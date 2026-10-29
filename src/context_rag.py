import json
import os
import re
from typing import TypedDict
from urllib.parse import quote
import requests
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from dotenv import load_dotenv

load_dotenv()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

WIKI_SEARCH_URL = "https://en.wikipedia.org/w/api.php"
WIKI_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"
REQUEST_HEADERS = {"User-Agent": "CulturalContextGraph/1.0 (educational project)"}
MAX_KEYWORDS = 4
MAX_WIKI_CHARS = 600
MAX_TAVILY_CHARS = 300

KEYWORD_SYSTEM_PROMPT = None
with open("src/context_rag_prompt/keyword_system_prompt.txt") as file:
    KEYWORD_SYSTEM_PROMPT = file.read()

ENRICH_SYSTEM_PROMPT = None
with open("src/context_rag_prompt/enrich_prompt.txt") as file:
    ENRICH_SYSTEM_PROMPT = file.read()


class PipelineState(TypedDict, total=False):
    user_query: str
    language: str
    translation: str
    keywords: list[str]
    context: dict
    final_prompt: str


def get_ollama_llm(json_mode=False):
    return ChatOllama(
        model=OLLAMA_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,
        num_ctx=8192,
        format="json" if json_mode else None,
    )


def parse_keyword_output(raw_text, fallback_translation):
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return "unknown", fallback_translation, []
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "unknown", fallback_translation, []
    keywords = []
    for keyword in data.get("keywords", []):
        if isinstance(keyword, str) and keyword.strip() and keyword.strip() not in keywords:
            keywords.append(keyword.strip())
    translation = data.get("translation")
    if not isinstance(translation, str) or not translation.strip():
        translation = fallback_translation
    return str(data.get("language", "unknown")), translation.strip(), keywords[:MAX_KEYWORDS]


def fetch_wikipedia(term):
    print(f"[Wikipedia] searching: {term}")
    try:
        search_response = requests.get(
            WIKI_SEARCH_URL,
            params={
                "action": "query",
                "list": "search",
                "srsearch": term,
                "format": "json",
                "srlimit": 1,
            },
            headers=REQUEST_HEADERS,
            timeout=10,
        )
        search_response.raise_for_status()
        hits = search_response.json()["query"]["search"]
        if not hits:
            print(f"[Wikipedia] no result for: {term}")
            return ""

        title = hits[0]["title"]
        summary_response = requests.get(
            WIKI_SUMMARY_URL.format(title=quote(title.replace(" ", "_"))),
            headers=REQUEST_HEADERS,
            timeout=10,
        )
        summary_response.raise_for_status()
        text = summary_response.json().get("extract", "")[:MAX_WIKI_CHARS]
        print(f"[Wikipedia] page '{title}' returned {len(text)} chars")
        return text
    except Exception as error:
        print(f"[Wikipedia] error for '{term}': {error}")
        return ""


def fetch_tavily(term):
    print(f"[Tavily] searching: {term}")
    if not TAVILY_API_KEY:
        print("[Tavily] TAVILY_API_KEY not set, skipping")
        return ""
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(
            query=f"what does {term} look like visual appearance decorations clothing",
            max_results=2,
        )
        snippets = [
            result["content"][:MAX_TAVILY_CHARS]
            for result in response.get("results", [])
            if result.get("content")
        ]
        print(f"[Tavily] {len(snippets)} snippets for '{term}'")
        return "\n".join(snippets)
    except Exception as error:
        print(f"[Tavily] error for '{term}': {error}")
        return ""


def format_context(context):
    blocks = []
    for term, sources in context.items():
        block = f"## {term}"
        if sources["wikipedia"]:
            block += f"\nWikipedia: {sources['wikipedia']}"
        if sources["tavily"]:
            block += f"\nWeb: {sources['tavily']}"
        blocks.append(block)
    return "\n\n".join(blocks)


def build_graph(keyword_llm=None, enrich_llm=None):
    keyword_llm = keyword_llm or get_ollama_llm(json_mode=True)
    enrich_llm = enrich_llm or get_ollama_llm()

    def keyword_node(state):
        query = state["user_query"]
        print(f"[KeywordAgent] query: {query}")
        response = keyword_llm.invoke(
            [SystemMessage(content=KEYWORD_SYSTEM_PROMPT), HumanMessage(content=query)]
        )
        language, translation, keywords = parse_keyword_output(response.content, query)
        print(f"[KeywordAgent] language: {language}, translation: {translation}, keywords: {keywords}")
        return {"language": language, "translation": translation, "keywords": keywords}

    def retrieve_node(state):
        context = {}
        for term in state["keywords"]:
            wikipedia_text = fetch_wikipedia(term)
            tavily_text = fetch_tavily(term)
            if not wikipedia_text and not tavily_text:
                print(f"[Retriever] nothing found for '{term}', dropping")
                continue
            context[term] = {"wikipedia": wikipedia_text, "tavily": tavily_text}
        print(f"[Retriever] context collected for: {list(context.keys())}")
        return {"context": context}

    def enrich_node(state):
        print("[EnrichmentAgent] combining query with context")
        # Request goes last so it can never be the part that gets truncated.
        human_message = (
            f"Research:\n{format_context(state['context'])}\n\n"
            f"Original request: {state['user_query']}\n"
            f"English translation: {state['translation']}"
        )
        response = enrich_llm.invoke(
            [SystemMessage(content=ENRICH_SYSTEM_PROMPT), HumanMessage(content=human_message)]
        )
        final_prompt = response.content.strip()
        print(f"[EnrichmentAgent] final prompt generated ({len(final_prompt)} chars)")
        return {"final_prompt": final_prompt}

    def passthrough_node(state):
        print("[Passthrough] no cultural context available, using original query")
        return {"final_prompt": state.get("translation") or state["user_query"]}

    def route_after_keywords(state):
        return "retrieve" if state.get("keywords") else "passthrough"

    def route_after_retrieval(state):
        return "enrich" if state.get("context") else "passthrough"

    graph = StateGraph(PipelineState)
    graph.add_node("keywords", keyword_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("enrich", enrich_node)
    graph.add_node("passthrough", passthrough_node)

    graph.add_edge(START, "keywords")
    graph.add_conditional_edges(
        "keywords", route_after_keywords, {"retrieve": "retrieve", "passthrough": "passthrough"}
    )
    graph.add_conditional_edges(
        "retrieve", route_after_retrieval, {"enrich": "enrich", "passthrough": "passthrough"}
    )
    graph.add_edge("enrich", END)
    graph.add_edge("passthrough", END)
    return graph.compile()


def run_pipeline(user_query, graph=None):
    graph = graph or build_graph()
    return graph.invoke({"user_query": user_query})


if __name__ == "__main__":
    user_query = "जयपुर के बाज़ार में एक चाय वाला, हाथ में केतली, पीछे रंग-बिरंगी दुकानें,"

    print(f"< ----- Running with model: {OLLAMA_MODEL} ----- >")
    result = run_pipeline(user_query)

    print(result["context"])

    print("\n< ----- Result ----- >")
    print(f"Language    : {result.get('language')}")
    print(f"Translation : {result.get('translation')}")
    print(f"Keywords    : {result.get('keywords')}")
    print(f"Context     : {list(result.get('context', {}).keys())}")
    print(f"Final prompt:\n{result['final_prompt']}")

    assert result.get("keywords"), "No keywords were extracted"
    assert result.get("context"), "No context was retrieved from Wikipedia or Tavily"
    assert result["final_prompt"] != user_query, "Prompt was not enriched"
    print("\n< ----- Pipeline check passed ----- >")


