from typing import Dict, Any, Optional
from src.indic_bert_embedder import IndicBERTEmbedder
from src.cultural_retriever import CulturalWebRetriever
from src.context_filter import CulturalContextFilter
from src.prompt_builder import StructuredPromptBuilder

class CulturalPromptPipeline:
    def __init__(self, embedder_model: Optional[str] = None, max_tokens: int = 150):
        print("[Pipeline] Initializing IndicBERT Embedder...")
        self.embedder = IndicBERTEmbedder(model_name=embedder_model)
        
        print("[Pipeline] Initializing Cultural Web Retriever...")
        self.retriever = CulturalWebRetriever()
        
        print("[Pipeline] Initializing IndicBERT Context Filter...")
        self.filter = CulturalContextFilter(embedder=self.embedder, max_tokens=max_tokens)
        
        print("[Pipeline] Initializing Structured Prompt Builder...")
        self.prompt_builder = StructuredPromptBuilder()
        print("[Pipeline] Pipeline ready.")

    def run(self, user_query: str, return_details: bool = False) -> Dict[str, Any]:
        english_query = self.retriever.translate_query_to_english(user_query)

        retrieved_context = self.retriever.retrieve_cultural_context(user_query)

        filtered_categories = self.filter.filter_and_categorize(user_query, retrieved_context)
        compressed_visual_tokens = self.filter.compress_to_visual_tokens(filtered_categories)

        result = self.prompt_builder.build_prompt(
            original_query=user_query,
            english_query=english_query,
            filtered_categories=filtered_categories
        )

        output = {
            "structured_prompt": result["full_prompt"],
            "negative_prompt": result["negative_prompt"],
            "prompt_blocks": result["structured_blocks"]
        }

        if return_details:
            output["intermediate"] = {
                "user_query": user_query,
                "english_query": english_query,
                "retrieved_context_sources": [c.get("source") for c in retrieved_context],
                "retrieved_raw_texts": [c.get("text") for c in retrieved_context],
                "filtered_categories": filtered_categories,
                "compressed_visual_tokens": compressed_visual_tokens,
                "indic_bert_model": self.embedder.model_name
            }

        return output
