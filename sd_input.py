import sys
import requests

API_URL = "http://127.0.0.1:8000/generate-prompt"

def get_sd_prompts_from_api(user_query: str):
    try:
        response = requests.post(
            API_URL,
            json={"query": user_query, "verbose": False}
        ).json()
        
        return response["structured_prompt"], response["negative_prompt"]

    except Exception as e:
        print(f"Error connecting to API ({API_URL}): {e}")
        print("Make sure the FastAPI server is running: ./venv/bin/python -m uvicorn api:app --reload")
        sys.exit(1)

