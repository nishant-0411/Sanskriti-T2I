import os
from typing import List, Dict
from dotenv import load_dotenv
load_dotenv()

# pyrefly: ignore [missing-import]
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.messages import SystemMessage, HumanMessage
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq

MODEL_NAME = "Qwen/Qwen3.5-9B"
PROMPT_DIR = "src/multi_agent_pipeline/prompts/social_agent"

def _load_prompt(filename: str) -> str:
    """Read a prompt template file (containing {placeholders}) from PROMPT_DIR."""
    path = os.path.join(PROMPT_DIR, filename)
    with open(path, "r") as file:
        return file.read()

def get_llm(temperature: float = 0.3, max_new_tokens: int = 600):
    """Build a ChatHuggingFace client for Qwen3.5-9B with thinking mode disabled."""
    print(f"[get_llm] Building HF endpoint client for model={MODEL_NAME}, temp={temperature}")
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    hf_endpoint = HuggingFaceEndpoint(
        model=MODEL_NAME,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        huggingfacehub_api_token=hf_token,
        model_kwargs={
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}
        },
    )
    llm = ChatHuggingFace(llm=hf_endpoint)
    print("[get_llm] HF ChatHuggingFace client ready")
    return llm


def get_groq_llm(temperature: float = 0.3, max_new_tokens: int = 600):
    print(f"[get_groq_llm] Building Groq client, temp={temperature}, max_tokens={max_new_tokens}")
    groq_key = os.getenv("GROQ_API_KEY")
    chat = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=temperature,
        max_tokens=max_new_tokens,
        groq_api_key=groq_key,
    )
    print("[get_groq_llm] Groq client ready")
    return chat


def agent2_llm(temperature: float = 0.3, max_new_tokens: int = 600):
    """Build a ChatHuggingFace client for openai/gpt-oss-20b."""
    print(f"[agent2_llm] Building HF endpoint client for model=openai/gpt-oss-20b, temp={temperature}")
    hf_endpoint = HuggingFaceEndpoint(
        model="openai/gpt-oss-20b",
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        
    )
    llm = ChatHuggingFace(llm=hf_endpoint)
    print("[agent2_llm] HF ChatHuggingFace client ready")
    return llm


def _call(llm, system_prompt: str, trigger: str = "Proceed.") -> str:
    """Helper to invoke the LLM with a system prompt + a minimal human trigger
    message (some chat templates require at least one user turn)."""
    print(f"[_call] Invoking LLM ({type(llm).__name__}) | system_prompt_len={len(system_prompt)} chars")
    messages = [SystemMessage(content=system_prompt), HumanMessage(content=trigger)]
    try:
        response = llm.invoke(messages)
    except Exception as e:
        print(f"[_call] ERROR during llm.invoke: {e}")
        raise
    result = response.content.strip()
    print(f"[_call] Got response ({len(result)} chars): {result[:100]!r}{'...' if len(result) > 100 else ''}")
    return result


class SocialAgent:
    """
    A single Social Agent that owns one persona (culture / age_gender / landmark)
    and participates in multi-round Q&A with other agents to refine its
    visual description of that persona.
    """

    def __init__(self, name: str, persona_type: str, persona_value: str, llm=None):
        self.name = name
        self.persona_type = persona_type
        self.persona_value = persona_value
        self.llm = llm or get_llm()
        self.description = ""
        self.conversation_log: List[Dict] = []
        print(f"[SocialAgent:{self.name}] Initialized | persona_type={persona_type}, persona_value={persona_value}")

    # ---------- 1. Initial description ----------
    def generate_initial_description(self, moderator_context: str) -> str:
        print(f"[SocialAgent:{self.name}] generate_initial_description() start")
        template = _load_prompt("initial_description_system.txt")
        system_prompt = template.format(
            persona_type=self.persona_type,
            persona_value=self.persona_value,
            moderator_context=moderator_context,
        )

        self.description = _call(self.llm, system_prompt)
        self.conversation_log.append({"type": "initial_description", "content": self.description})
        print(f"[SocialAgent:{self.name}] generate_initial_description() done")
        return self.description

    # ---------- 2. Ask a question to another agent ----------
    def ask_question(self, other_agent: "SocialAgent") -> str:
        print(f"[SocialAgent:{self.name}] ask_question() -> target={other_agent.name}")
        template = _load_prompt("ask_question_system.txt")
        system_prompt = template.format(
            persona_type=self.persona_type,
            persona_value=self.persona_value,
            other_persona_type=other_agent.persona_type,
            other_persona_value=other_agent.persona_value,
            other_description=other_agent.description,
        )

        question = _call(self.llm, system_prompt)
        self.conversation_log.append({"type": f"question_to_{other_agent.name}", "content": question})
        print(f"[SocialAgent:{self.name}] ask_question() done -> {question[:80]!r}")
        return question

    # ---------- 3. Answer a question from another agent ----------
    def answer_question(self, question: str, asker_name: str) -> str:
        print(f"[SocialAgent:{self.name}] answer_question() <- from={asker_name}")
        template = _load_prompt("answer_question_system.txt")
        system_prompt = template.format(
            persona_type=self.persona_type,
            persona_value=self.persona_value,
            description=self.description,
            asker_name=asker_name,
            question=question,
        )

        answer = _call(self.llm, system_prompt)
        self.conversation_log.append({"type": f"answer_to_{asker_name}", "content": answer})
        print(f"[SocialAgent:{self.name}] answer_question() done -> {answer[:80]!r}")
        return answer

    # ---------- 4. Refine description using new Q&A info ----------
    def refine_description(self, new_info: str) -> str:
        print(f"[SocialAgent:{self.name}] refine_description() start | new_info={new_info[:80]!r}")
        template = _load_prompt("redefine_description.txt")
        system_prompt = template.format(
            persona_type=self.persona_type,
            persona_value=self.persona_value,
            description=self.description,
            new_info=new_info,
        )

        self.description = _call(self.llm, system_prompt)
        self.conversation_log.append({"type": "refined_description", "content": self.description})
        print(f"[SocialAgent:{self.name}] refine_description() done")
        return self.description


class SocialAgentOrchestrator:
    """
    Coordinates multiple Social Agents through multi-round Q&A conversations
    and merges their refined descriptions into one comprehensive image caption.
    """

    def __init__(self, agents: List[SocialAgent], rounds: int = 2, llm=None):
        self.agents = agents
        self.rounds = rounds
        self.llm = llm or get_llm()
        print(f"[Orchestrator] Initialized with {len(agents)} agents, rounds={rounds}")

    def run(self, moderator_context: str):
        print(f"[Orchestrator] run() start | context={moderator_context[:80]!r}")

        # Step 1: each agent generates an initial description
        print("[Orchestrator] Step 1: generating initial descriptions for all agents")
        for agent in self.agents:
            agent.generate_initial_description(moderator_context)
            print(f"< --- {agent.name} Intial Description --- >")
            print(agent.conversation_log)
            
        print("[Orchestrator] Step 1 complete")

        # Step 2: multi-round question-answering between every ordered pair
        print(f"[Orchestrator] Step 2: starting {self.rounds} round(s) of Q&A")
        for round_num in range(self.rounds):
            print(f"[Orchestrator] --- Round {round_num + 1}/{self.rounds} ---")
            for asker in self.agents:
                for answerer in self.agents:
                    if asker is answerer:
                        continue
                    print(f"[Orchestrator] Pair: asker={asker.name} -> answerer={answerer.name}")
                    question = asker.ask_question(answerer)
                    answer = answerer.answer_question(question, asker.name)
                    asker.refine_description(f"{answerer.name} said: {answer}")
            print(f"[Orchestrator] --- Round {round_num + 1}/{self.rounds} complete ---")

        print("[Orchestrator] run() complete")

        # Return each agent's final refined description
        return {agent.name: agent.description for agent in self.agents}


if __name__ == "__main__":
    print("[main] Starting pipeline")

    agent1 = SocialAgent(
        name="agent-1",
        persona_type="age_gender",
        persona_value="30-year-old woman",
        llm=get_groq_llm(),
    )

    agent2 = SocialAgent(
        name="agent-2",
        persona_type="location",
        persona_value="patna",
        llm=get_groq_llm(),
    )

    agent3 = SocialAgent(
        name="agent-3",
        persona_type="festival",
        persona_value="Chhath Puja",
        llm=get_groq_llm(),
    )

    test_query = (
        "During Chhath Puja celebrations in Patna, a 30-year-old woman offered "
        "sunset arghya at the river. Her elderly grandmother, aged around 70, "
        "watched from the ghat steps."
    )

    print("[main] Agents created, starting orchestrator")
    orchestrator = SocialAgentOrchestrator([agent1, agent2, agent3], rounds=1)
    try:
        final_descriptions = orchestrator.run(moderator_context=test_query)
        print("[main] Pipeline finished successfully")
        for name, desc in final_descriptions.items():
            print(f"\n=== {name} final description ===\n{desc}")
    except Exception as e:
        print(f"[main] Pipeline FAILED with error: {e}")
        raise e