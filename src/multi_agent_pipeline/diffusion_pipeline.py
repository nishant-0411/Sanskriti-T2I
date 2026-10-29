from src.multi_agent_pipeline.moderator_agent import recognize_entity
from src.multi_agent_pipeline.social_agent import SocialAgent, SocialAgentOrchestrator, get_groq_llm, get_ollama_llm
from src.multi_agent_pipeline.summarizer_agent import SummarizingAgent, generate_image
from src.multi_agent_pipeline.prompt_refinement import refine_prompt_from_image
from src.context_rag import run_pipeline
from pathlib import Path

THRESHOLD = 7

# Pipeline 1
def run(user_query: str, output_path = "image2.png", verbose: bool = False):
    # Fetch The Information of culture
    state = run_pipeline(user_query)
    context_query = state["final_prompt"]

    # Moderator Agent
    print("< ----- Running Moderating Agent ----- >")
    result = recognize_entity(user_query=context_query)
    parsed = json.loads(result.content) # type: ignore

    # Social Agent Iteration
    print("< ---- Running Social Agent ---- >")
    social_agents = []
    for name, persona in parsed.items():
        if persona:
            social_agent = SocialAgent(name=f"{name} Agent", 
                                    persona_type=name, 
                                    persona_value=persona[0], 
                                    llm=get_groq_llm(),
                                    verbose=verbose
                                    )
            social_agents.append(social_agent)

    orechestrate = SocialAgentOrchestrator(social_agents, rounds=1, verbose=verbose)
    final_description = orechestrate.run(moderator_context=user_query)
    print(final_description)


    # Summaarization Agent and Image Generation
    print("< ---- Running Summarizing Agent ---- >")
    summarize = SummarizingAgent(llm=get_groq_llm())
    summarized_descritpion = summarize.summarize(final_description)

    print(summarized_descritpion)

    print("< ----- Generating Image ----- >")

# Pipeline 2
def run2(user_query: str, output_path: str = "test1/image.png", turns: int = 1):
    # Fetch The Information of culture
    print("< ---- Fetching Info From internet and Wikipedia ---- >")
    state = run_pipeline(user_query)
    context_query = state["final_prompt"]

    # Generate Image
    print("< ---- Generating First Image ---- >")
    image_path = generate_image(context_query, output_path=output_path)

    # Run The Prompt Refinement
    for _ in range(turns):
        print("< ---- Running Promt Refinement ---- >")

        state = refine_prompt_from_image(
            image_path=image_path,
            user_query=user_query,
            context_query=context_query
        )

        refined_prompt = state.get("refined_prompt")
        score: int = state.get("cultural_score") # type: ignore

        print("< ---- Generating Image ---- >")
        image_path = generate_image(refined_prompt, output_path=image_path) # type: ignore
        
        if isinstance(score, int) and score >= THRESHOLD:
            break


if __name__ == "__main__":
    # ---- Test case 2 (new): different festival, location, and persona ----
    test_query_2 = (
        "On Pongal day in a Tamil Nadu village, women draw kolam at the doorstep and cook pongal in a clay pot."
    )

    run2(test_query_2)



