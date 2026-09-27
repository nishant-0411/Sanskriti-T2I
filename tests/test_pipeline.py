import pytest
from src.indic_bert_embedder import IndicBERTEmbedder
from src.cultural_retriever import CulturalWebRetriever
from src.context_filter import CulturalContextFilter
from src.prompt_builder import StructuredPromptBuilder
from src.pipeline import CulturalPromptPipeline


def test_indic_bert_embedder():
    embedder = IndicBERTEmbedder()
    # Test single and batch encoding
    emb1 = embedder.encode("छठ पूजा")
    assert emb1 is not None
    assert emb1.shape[1] == 768 or emb1.shape[1] > 0

    emb_batch = embedder.encode(["दुर्गा पूजा", "पोंगल त्योहार"])
    assert emb_batch.shape[0] == 2

    # Test similarity score
    sim = embedder.compute_similarity("छठ पूजा", "सूर्य उपासना")
    assert isinstance(sim, float)
    assert sim > -1.0 and sim <= 1.0


def test_cultural_retriever():
    retriever = CulturalWebRetriever()
    query = "छठ पूजा अर्घ्य"
    
    # Test translation & keyword extraction
    terms = retriever.extract_search_terms(query)
    assert len(terms) > 0

    # Test retrieval
    context = retriever.retrieve_cultural_context(query)
    assert len(context) > 0
    assert "text" in context[0]


def test_context_filter():
    embedder = IndicBERTEmbedder()
    filter_obj = CulturalContextFilter(embedder=embedder, max_tokens=100)

    sample_context = [{
        "title": "Chhath Puja",
        "text": "Devotees stand in river waters to offer sunset arghya using bamboo soop filled with fruits and sugarcane. Women wear yellow sarees with vermilion nose markings. The population of Patna is 2 million.",
        "visual_keywords": ["soop", "yellow saree", "sindoor"]
    }]

    categorized = filter_obj.filter_and_categorize("छठ पूजा", sample_context)
    assert isinstance(categorized, dict)
    assert "attire_jewelry" in categorized
    assert "objects_artifacts" in categorized

    # Ensure non-visual statistical content (e.g., population stats) is pruned
    all_details = " ".join([d for details in categorized.values() for d in details])
    assert "population" not in all_details.lower()

    compressed = filter_obj.compress_to_visual_tokens(categorized)
    assert isinstance(compressed, str)


def test_prompt_builder():
    builder = StructuredPromptBuilder()
    filtered = {
        "attire_jewelry": ["yellow cotton saree", "sindoor nose marking"],
        "objects_artifacts": ["bamboo soop", "sugarcane"],
        "architecture_setting": ["riverbank ghat"]
    }
    result = builder.build_prompt("छठ पूजा", "Chhath Puja", filtered)
    assert "full_prompt" in result
    assert "negative_prompt" in result
    assert "Yellow cotton saree" in result["full_prompt"] or "yellow cotton saree" in result["full_prompt"]


def test_full_pipeline():
    pipeline = CulturalPromptPipeline()
    
    # Test Hindi query
    res_hi = pipeline.run("छठ पूजा अर्घ्य देती महिला")
    assert res_hi["structured_prompt"] is not None
    assert len(res_hi["structured_prompt"]) > 20

    # Test Bengali query
    res_bn = pipeline.run("দুর্গাপূজায় ধুনুচি নাচ")
    assert res_bn["structured_prompt"] is not None

    # Test Tamil query
    res_ta = pipeline.run("பொங்கல் திருநாள் கொண்டாட்டம்")
    assert res_ta["structured_prompt"] is not None

    print("\n✅ All Pipeline Unit & Integration Tests Passed!")

if __name__ == "__main__":
    test_indic_bert_embedder()
    test_cultural_retriever()
    test_context_filter()
    test_prompt_builder()
    test_full_pipeline()
