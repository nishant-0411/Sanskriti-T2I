from src.multi_agent_pipeline.moderator_agent import recognize_entity
from src.multi_agent_pipeline.social_agent import SocialAgent, SocialAgentOrchestrator, get_groq_llm
from src.multi_agent_pipeline.summarizer_agent import SummarizingAgent, generate_image
from src.pipeline import CulturalPromptPipeline
import json


def run_pipeline(user_query: str, output_path = "image2.png"):
    # Moderator Agent
    print("< ----- Running Moderating Agent ----- >")
    result = recognize_entity(user_query=user_query)
    parsed = json.loads(result.content) # type: ignore

    print(parsed)

    # Social Agent Iteration
    print("< ---- Running Social Agent ---- >")
    social_agents = []
    for name, persona in parsed.items():
        social_agent = SocialAgent(name=f"{name} Agent", 
                                   persona_type=name, 
                                   persona_value=persona[0], 
                                   llm=get_groq_llm()
                                )
        social_agents.append(social_agent)

    orechestrate = SocialAgentOrchestrator(social_agents, rounds=1)
    final_description = orechestrate.run(moderator_context=user_query)
    print(final_description)


    # Summaarization Agent and Image Generation
    print("< ---- Running Summarizing Agent ---- >")
    summarize = SummarizingAgent(llm=get_groq_llm())
    summarized_descritpion = summarize.summarize(final_description)

    print(summarized_descritpion)

    print("< ----- Generating Image ----- >")

    generate_image(summarized_descritpion, output_path=output_path)

def run_pipeline_2(user_query: str, output_path="image.png"):

    # Creating Pipeline 
    prompt_pipeline = CulturalPromptPipeline()
    final_response = prompt_pipeline.run(user_query)

    # generate image
    generate_image(final_response["structured_prompt"], output_path=output_path)

def run_pipeline_3():
    generate_image("An elderly man in a crisp white dhoti and kurta kneels in prayer with hands clasped before a towering, brilliantly lit pandal adorned with gold and crimson cloth, marigold garlands, and glowing lanterns, surrounded by a subtle crowd in evening festive ambience." , output_path="pandal.png")


if __name__ == "__main__":
    # ---- Test case 2 (new): different festival, location, and persona ----
    test_query_2 = (
        
    )

    
    run_pipeline_3()


