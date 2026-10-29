import json
import re
from pathlib import Path
from typing import Optional, TypedDict

import ollama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

TEXT_MODEL = "qwen2.5:3b"
VISION_MODEL = "qwen3-vl:2b"

llm_json = ChatOllama(model=TEXT_MODEL, temperature=0.2, format="json")

DIMENSIONS = ["attire", "architecture", "objects", "rituals", "setting", "symbolism", "authenticity"]
WEIGHTS = {d: 1.0 for d in DIMENSIONS}


# ---------------- state ----------------
class RefineState(TypedDict):
    image_path: str
    user_query: str
    context_query: str
    out_dir: Optional[str]
    
    description: str
    audit: dict
    
    refined_prompt: str
    cultural_score: float


def initial_state(image_path: str, user_query: str, context_query: str,
                  out_dir: Optional[str] = None) -> RefineState:
    """Every key is initialized, so the graph never sees a missing key."""
    return RefineState(
        image_path=str(image_path),
        user_query=user_query,
        context_query=context_query,
        out_dir=out_dir,
        description="",
        audit={},
        refined_prompt="",
        cultural_score=0.0,
    )


# ---------------- helpers ----------------
def _parse_json(text) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        try:
            return json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            return {}


def _ask_json(system: str, user: str, retries: int = 2) -> dict:
    for _ in range(retries + 1):
        out = llm_json.invoke([SystemMessage(content=system), HumanMessage(content=user)])
        data = _parse_json(out.content)
        if data:
            return data
    return {}


def _aggregate(scores: dict) -> float:
    num = den = 0.0
    for d in DIMENSIONS:
        v = scores.get(d)
        if isinstance(v, (int, float)):
            num += WEIGHTS[d] * v
            den += WEIGHTS[d]
    return round(num / den, 2) if den else 0.0


# ---------------- nodes ----------------
DESCRIBE_SYSTEM = (
    "Describe this image objectively and in detail. Do NOT judge it. "
    "List what you see: people and their clothing/jewelry, buildings and decor, "
    "objects/food/instruments, activities and gestures, setting, colors and symbols. "
    "Be literal. If something is unclear, say it is unclear."
)


def describe_node(state: RefineState) -> dict:
    resp = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {"role": "system", "content": DESCRIBE_SYSTEM},
            {"role": "user", "content": "Describe the image.", "images": [state["image_path"]]},
        ],
        options={"temperature": 0.1},
    )
    return {"description": resp["message"]["content"]}


JUDGE_SYSTEM = f"""You are an expert in the cultural context given by the user.
You receive the user's original query, the context query (the prompt used to generate
the image, which contains the cultural grounding), and an objective description of the
generated image. Judge how culturally CORRECT the image is for that context.

Score each dimension 1-10 (10 = authentic and accurate, 1 = wrong or absent):
{", ".join(DIMENSIONS)}.

Rules:
- Only flag problems the description actually supports. Do not invent details.
- Penalize elements from another culture or region, anachronisms, costume-party
  stereotypes, and generic "exotic" looks.
- If a dimension cannot be judged from the description, use null.
- Every issue needs a concrete correction: what it SHOULD look like.

Respond ONLY with JSON:
{{"scores": {{{", ".join(f'"{d}": <int|null>' for d in DIMENSIONS)}}},
  "correct_elements": ["..."],
  "issues": [{{"dimension": "...", "problem": "...", "correction": "..."}}],
  "overall_comment": "..."}}"""


def judge_node(state: RefineState) -> dict:
    user = (
        f"User query:\n\"\"\"{state['user_query']}\"\"\"\n\n"
        f"Context query used for generation:\n\"\"\"{state['context_query']}\"\"\"\n\n"
        f"Objective image description:\n\"\"\"{state['description']}\"\"\"\n\n"
        "Evaluate cultural correctness."
    )
    audit = _ask_json(JUDGE_SYSTEM, user)
    audit.setdefault("scores", {})
    audit.setdefault("correct_elements", [])
    audit.setdefault("issues", [])
    audit.setdefault("overall_comment", "")
    audit["cultural_score"] = _aggregate(audit["scores"])
    return {"audit": audit, "cultural_score": audit["cultural_score"]}


REFINE_SYSTEM = """You rewrite text-to-image prompts to make them culturally accurate.
Input: the user's original query, the current context query, and a cultural audit of the
image it produced. Write a new context query that:
- stays faithful to the user's original intent,
- keeps every culturally correct element,
- fixes each issue by stating the correction explicitly (named garments, textiles,
  architecture, objects, colors, rituals, setting),
- adds short negative constraints where useful (e.g. "no Western-style gown"),
- is one descriptive paragraph under ~130 words, with no meta commentary.
Respond ONLY with JSON: {"refined_prompt": "<text>"}"""


def refine_node(state: RefineState) -> dict:
    audit = state["audit"]
    user = (
        f"User query:\n{state['user_query']}\n\n"
        f"Current context query:\n{state['context_query']}\n\n"
        "Cultural audit:\n"
        + json.dumps({k: audit[k] for k in ("scores", "correct_elements", "issues")}, indent=2)
    )
    out = _ask_json(REFINE_SYSTEM, user)
    refined = (out.get("refined_prompt") or "").strip()
    return {"refined_prompt": refined or state["context_query"]}  # fall back to original


def save_node(state: RefineState) -> dict:
    if state["out_dir"]:
        out = Path(state["out_dir"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "refined_prompt.txt").write_text(state["refined_prompt"], encoding="utf-8")
        (out / "refinement.json").write_text(
            json.dumps({
                "user_query": state["user_query"],
                "context_query": state["context_query"],
                "image_path": state["image_path"],
                "description": state["description"],
                "audit": state["audit"],
                "refined_prompt": state["refined_prompt"],
            }, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return {}


# ---------------- graph ----------------
def build_refiner():
    g = StateGraph(RefineState)
    g.add_node("describe", describe_node)
    g.add_node("judge", judge_node)
    g.add_node("refine", refine_node)
    g.add_node("save", save_node)

    g.add_edge(START, "describe")
    g.add_edge("describe", "judge")
    g.add_edge("judge", "refine")
    g.add_edge("refine", "save")
    g.add_edge("save", END)
    return g.compile()


_refiner = build_refiner()


# ---------------- Component Function ----------------
def refine_prompt_from_image(image_path: str, user_query: str, context_query: str,
                             out_dir: Optional[str] = "test1") -> dict:
    """Single refinement pass: image + queries in, refined context query + audit out."""
    final = _refiner.invoke(initial_state(image_path, user_query, context_query, out_dir))
    return {
        "refined_prompt": final["refined_prompt"],
        "cultural_score": final["cultural_score"],
        "audit": final["audit"],
    }
