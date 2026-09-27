from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import uvicorn
from src.pipeline import CulturalPromptPipeline

app = FastAPI(
    title="Multilingual Cultural Prompt-Generation API",
    description="REST API for generating structured, culturally grounded image-generation prompts from Indic language queries.",
    version="1.0.0"
)

pipeline_instance: Optional[CulturalPromptPipeline] = None

def get_pipeline():
    global pipeline_instance
    if pipeline_instance is None:
        print("[API] Initializing Cultural Prompt Pipeline...")
        pipeline_instance = CulturalPromptPipeline()
    return pipeline_instance

class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        description="User query in any Indian language or English",
        example="छठ पूजा अर्घ्य देती महिला"
    )
    verbose: bool = Field(
        default=True,
        description="Whether to include intermediate retrieval & IndicBERT filtering details in the response"
    )


class QueryResponse(BaseModel):
    status: str
    query: str
    english_concept: str
    structured_prompt: str
    negative_prompt: str
    prompt_blocks: Dict[str, str]
    filtered_visual_categories: Optional[Dict[str, List[str]]] = None
    retrieved_sources: Optional[List[str]] = None


@app.get("/")
def root():
    """Health check & API info endpoint."""
    return {
        "status": "online",
        "service": "Multilingual Cultural Prompt-Generation API",
        "swagger_docs": "http://127.0.0.1:8000/docs",
        "redoc_docs": "http://127.0.0.1:8000/redoc"
    }


@app.post("/generate-prompt", response_model=QueryResponse)
def generate_prompt_endpoint(request: QueryRequest):
    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    pipeline = get_pipeline()
    result = pipeline.run(request.query, return_details=True)

    inter = result.get("intermediate", {})

    return QueryResponse(
        status="success",
        query=request.query,
        english_concept=inter.get("english_query", request.query),
        structured_prompt=result.get("structured_prompt", ""),
        negative_prompt=result.get("negative_prompt", ""),
        prompt_blocks=result.get("prompt_blocks", {}),
        filtered_visual_categories=inter.get("filtered_categories") if request.verbose else None,
        retrieved_sources=inter.get("retrieved_context_sources") if request.verbose else None
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
