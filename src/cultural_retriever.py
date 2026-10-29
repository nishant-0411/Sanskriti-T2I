import re
import requests
from typing import List, Dict, Any, Optional
import wikipedia
from deep_translator import GoogleTranslator

class CulturalWebRetriever:
    CULTURAL_KNOWLEDGE_BASE = {

    }

    INDIC_TERM_MAP = {
        "छठ": "Chhath Puja",
        "पूजा": "Puja ceremony",
        "अर्घ्य": "Arghya river water offering",
        "देती": "offering",
        "देते": "offering",
        "करती": "performing",
        "करते": "performing",
        "मनाती": "celebrating",
        "उत्सव": "festive celebration",
        "महिला": "woman",
        "महिलाएँ": "women",
        "स्त्री": "woman",
        "दुर्गा": "Durga Puja",
        "धुनुचि": "Dhunuchi dance",
        "नाच": "dance",
        "नृत्य": "dance",
        "பொங்கல்": "Pongal festival",
        "கொண்டாட்டம்": "celebration",
        "సంక్రాంతి": "Sankranti festival",
        "భోగి": "Bhogi bonfire",
        "गुढीपाडवा": "Gudi Padwa",
        "मिरवणूक": "procession"
    }

    def __init__(self, lang: str = "en"):
        self.default_lang = lang
        wikipedia.set_lang(lang)

    def translate_query_to_english(self, query: str) -> str:
        mapped_words = []
        for word in query.split():
            clean_w = word.strip()
            if clean_w in self.INDIC_TERM_MAP:
                mapped_words.append(self.INDIC_TERM_MAP[clean_w])
            else:
                mapped_words.append(clean_w)
                
        mapped_phrase = " ".join(mapped_words)
        
        try:
            translated = GoogleTranslator(source='auto', target='en').translate(query)
            if translated and len(translated) > 2:
                return translated
        except Exception:
            pass
            
        return mapped_phrase

    def extract_search_terms(self, query: str) -> List[str]:
        translated = self.translate_query_to_english(query)
        terms = [query, translated]
        
        clean_en = re.sub(r'[^\w\s]', '', translated.lower())
        words = clean_en.split()
        
        for i in range(len(words)):
            terms.append(words[i])
            if i + 1 < len(words):
                terms.append(f"{words[i]} {words[i+1]}")
                
        seen = set()
        unique_terms = []
        for t in terms:
            t_strip = t.strip()
            if t_strip and t_strip.lower() not in seen:
                seen.add(t_strip.lower())
                unique_terms.append(t_strip)
                
        return unique_terms

    def fetch_term_using_llm(self, query: str) -> List[str]:
        return ["Not ready yet"]

    def fetch_wikipedia_content(self, search_term: str) -> Optional[Dict[str, str]]:
        try:
            search_results = wikipedia.search(search_term)
            if not search_results:
                return None
            
            page_title = search_results[0]
            page = wikipedia.page(page_title, auto_suggest=False)
            
            return {
                "title": page.title,
                "summary": page.summary,
                "content": page.content[:3000],
                "url": page.url
            }
        except Exception:
            return None

    def retrieve_cultural_context(self, user_query: str) -> List[Dict[str, Any]]:
        results = []
        query_lower = user_query.lower()
        search_terms = self.extract_search_terms(user_query)
        
        for key, data in self.CULTURAL_KNOWLEDGE_BASE.items():
            if key in query_lower or any(key in term.lower() for term in search_terms):
                results.append({
                    "source": f"Curated Knowledge Base ({data['title']})",
                    "title": data['title'],
                    "text": data['en_summary'],
                    "visual_keywords": data['visual_keywords']
                })

        print(search_terms)
        
        if not results:
            # Retrive Entity for searching 
            wiki_fetched_count = 0
            

        if not results:
            translated = self.translate_query_to_english(user_query)
            results.append({
                "source": "Synthesized Regional Cultural Retrieval",
                "title": f"Cultural context for {translated}",
                "text": f"Traditional Indian cultural celebration depicting {translated} with authentic regional attire, brass accessories, ritual offerings, floral decorations, and architectural background motifs."
            })
            
        return results
