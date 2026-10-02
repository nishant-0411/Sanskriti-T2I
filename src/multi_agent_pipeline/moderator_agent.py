import os
import json
from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate

# Configurations
MODEL_NAME = "Qwen/Qwen3.5-9B"

# < ---- Recognize varies entity in Prompt like location, age, gender ---->
def recognize_entity(user_query: str):

    # Initializing Hugging Face API
    hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN") or os.getenv("HF_TOKEN")
    hf_endpoint = HuggingFaceEndpoint(
        model=MODEL_NAME,
        temperature=0.2,
        max_new_tokens=2000,
        huggingfacehub_api_token=hf_token,
        model_kwargs={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
    )

    llm = ChatHuggingFace(llm=hf_endpoint, verbose=True)

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


    # Expected (roughly):
    # age_gender: ["30-year-old woman", "elderly grandmother, aged around 70"] (wording may vary)
    # location: ["Patna"]
    # festival: ["Chhath Puja"]

    result = recognize_entity(user_query=test_query)
    print("RAW OUTPUT:\n", result)

    # Sanity check — does it even parse as JSON?
    try:
        parsed = json.loads(result.content)
        print("\n✅ Valid JSON. Parsed output:")
        print(json.dumps(parsed, indent=2))
    except json.JSONDecodeError as e:
        print("\n❌ Model did not return valid JSON:", e)







