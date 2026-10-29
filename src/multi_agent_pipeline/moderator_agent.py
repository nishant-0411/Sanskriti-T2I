import os
import json
from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import ChatOllama
from src.multi_agent_pipeline.social_agent import get_groq_llm
from langchain_core.prompts import PromptTemplate

# Configurations
MODEL_NAME = "qwen2.5:3b"

# < ---- Recognize varies entity in Prompt like location, age, gender ---->
def recognize_entity(user_query: str):

    # Initializing Ollama Model
    llm = get_groq_llm()

    # Loading The Prompt
    with open("src/multi_agent_pipeline/prompts/moderator_agent_prompt.txt", "r") as file:
        prompt = file.read()

    prompt_template = PromptTemplate(template=prompt, input_variables=["input_text"])


    # Bulding the chain 
    chain = prompt_template | llm

    response = chain.invoke({"input_text": user_query})

    return response

if __name__ == "__main__":
    test_query = (
        "During Chhath Puja celebrations in Patna, a 30-year-old woman offered "
        "sunset arghya at the river. Her elderly grandmother, aged around 70, "
        "watched from the ghat steps."
    )


    result = recognize_entity(user_query=test_query)
    print("RAW OUTPUT:\n", result)

    # Sanity check — does it even parse as JSON?
    try:
        parsed = json.loads(result.content)
        print("\n✅ Valid JSON. Parsed output:")
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError as e:
        print("\n❌ Model did not return valid JSON:", e)







