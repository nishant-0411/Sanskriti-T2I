import os
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

MODEL_NAME = "Qwen/Qwen3.5-9B"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
PROMPT_DIR = "src/multi_agent_pipeline/prompts/social_agent"


def _load_prompt(filename: str) -> str:
    """Read a prompt template file (containing {placeholders}) from PROMPT_DIR."""
    with open(os.path.join(PROMPT_DIR, filename), "r", encoding="utf-8") as file:
        return file.read()


def get_ollama_llm(temperature: float = 0.3, max_new_tokens: int = 600):
    """Return a ChatOllama client."""
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        num_ctx=4096,
        num_predict=max_new_tokens,
        keep_alive="10m",
    )


def get_llm(temperature: float = 0.3, max_new_tokens: int = 600):
    """Build a ChatHuggingFace client for Qwen3.5-9B with thinking mode disabled."""
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    hf_endpoint = HuggingFaceEndpoint(
        model=MODEL_NAME,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        huggingfacehub_api_token=hf_token,
        model_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    )
    return ChatHuggingFace(llm=hf_endpoint)


def get_groq_llm(temperature: float = 0.3, max_new_tokens: int = 600):
    return ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=temperature,
        max_tokens=max_new_tokens,
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )


def agent2_llm(temperature: float = 0.3, max_new_tokens: int = 600):
    """Build a ChatHuggingFace client for openai/gpt-oss-20b."""
    hf_endpoint = HuggingFaceEndpoint(
        model="openai/gpt-oss-20b",
        temperature=temperature,
        max_new_tokens=max_new_tokens,
    )
    return ChatHuggingFace(llm=hf_endpoint)


def _call(llm, system_prompt: str, trigger: str = "Proceed.") -> str:
    """Invoke the LLM with a system prompt plus a minimal human turn
    (some chat templates require at least one user message)."""
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=trigger)]
    return llm.invoke(messages).content.strip()


class SocialAgent:
    """
    One Social Agent that owns a persona (culture / age_gender / landmark)
    and refines its visual description through Q&A with other agents.
    """

    def __init__(self, name: str, persona_type: str, persona_value: str, llm=None, verbose: bool = False):
        self.name = name
        self.persona_type = persona_type
        self.persona_value = persona_value
        self.llm = llm or get_llm()
        self.description = ""
        self.conversation_log: List[Dict] = []
        self.verbose = verbose

    def _log(self, message: str):
        if self.verbose:
            print(f"[{self.name}] {message}")

    def generate_initial_description(self, moderator_context: str) -> str:
        system_prompt = _load_prompt("initial_description_system.txt").format(
            persona_type=self.persona_type,
            persona_value=self.persona_value,
            moderator_context=moderator_context,
        )
        self.description = _call(self.llm, system_prompt)
        self.conversation_log.append({"type": "initial_description", "content": self.description})
        self._log(f"initial description: {self.description[:80]!r}")
        return self.description

    def ask_question(self, other_agent: "SocialAgent") -> str:
        system_prompt = _load_prompt("ask_question_system.txt").format(
            persona_type=self.persona_type,
            persona_value=self.persona_value,
            other_persona_type=other_agent.persona_type,
            other_persona_value=other_agent.persona_value,
            other_description=other_agent.description,
        )
        question = _call(self.llm, system_prompt)
        self.conversation_log.append({"type": f"question_to_{other_agent.name}", "content": question})
        self._log(f"asked {other_agent.name}: {question[:80]!r}")
        return question

    def answer_question(self, question: str, asker_name: str) -> str:
        system_prompt = _load_prompt("answer_question_system.txt").format(
            persona_type=self.persona_type,
            persona_value=self.persona_value,
            description=self.description,
            asker_name=asker_name,
            question=question,
        )
        answer = _call(self.llm, system_prompt)
        self.conversation_log.append({"type": f"answer_to_{asker_name}", "content": answer})
        self._log(f"answered {asker_name}: {answer[:80]!r}")
        return answer

    def refine_description(self, new_info: str) -> str:
        system_prompt = _load_prompt("redefine_description.txt").format(
            persona_type=self.persona_type,
            persona_value=self.persona_value,
            description=self.description,
            new_info=new_info,
        )
        self.description = _call(self.llm, system_prompt)
        self.conversation_log.append({"type": "refined_description", "content": self.description})
        self._log("description refined")
        return self.description


class SocialAgentOrchestrator:
    """Runs multi-round Q&A between Social Agents and returns their refined descriptions."""

    def __init__(self, agents: List[SocialAgent], rounds: int = 1, verbose: bool = False):
        self.agents = agents
        self.rounds = rounds
        self.verbose = verbose

    def run(self, moderator_context: str) -> Dict[str, str]:
        for agent in self.agents:
            agent.generate_initial_description(moderator_context)

        for round_num in range(self.rounds):
            if self.verbose:
                print(f"[Orchestrator] round {round_num + 1}/{self.rounds}")
            for asker in self.agents:
                for answerer in self.agents:
                    if asker is answerer:
                        continue
                    question = asker.ask_question(answerer)
                    answer = answerer.answer_question(question, asker.name)
                    asker.refine_description(f"{answerer.name} said: {answer}")

        return {agent.name: agent.description for agent in self.agents}


if __name__ == "__main__":
    llm = get_ollama_llm()

    agents = [
        SocialAgent("agent-1", "age_gender", "30-year-old woman", llm=llm),
        SocialAgent("agent-2", "location", "patna", llm=llm),
        SocialAgent("agent-3", "festival", "Chhath Puja", llm=llm),
    ]

    test_query = (
        "During Chhath Puja celebrations in Patna, a 30-year-old woman offered "
        "sunset arghya at the river. Her elderly grandmother, aged around 70, "
        "watched from the ghat steps."
    )

    final_descriptions = SocialAgentOrchestrator(agents, rounds=1).run(test_query)
    for name, desc in final_descriptions.items():
        print(f"\n=== {name} final description ===\n{desc}")