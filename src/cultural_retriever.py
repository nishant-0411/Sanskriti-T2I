import re
import requests
from typing import List, Dict, Any, Optional
import wikipedia
from deep_translator import GoogleTranslator

class CulturalWebRetriever:
    CULTURAL_KNOWLEDGE_BASE = {
        "chhath": {
            "title": "Chhath",
            "en_summary": "Chhath is an ancient Hindu festival dedicated to the Sun God Surya and Chhathi Maiya. The central visual centerpiece is the glowing setting sun (Surya Dev) reflecting over sacred river waters as devotees perform rituals on riverbanks (ghats) at sunrise and sunset, offering arghya (water/milk offerings) using bamboo winnowing trays (soop) filled with sugarcane, seasonal fruits, and wheat-flour sweets (thekua). Women wear traditional yellow/orange unstitched cotton or silk sarees with vermilion (sindoor) applied from the nose tip to the hairline.",
            "visual_keywords": ["Sun God Surya Dev centerpiece", "riverbank ghat sunset arghya", "soop", "thekua", "ghat", "riverbank", "sindoor", "sugarcane", "yellow saree", "brass thali"]
        },
        "pongal": {
            "title": "Pongal (festival)",
            "en_summary": "Pongal is a multi-day harvest festival celebrated in Tamil Nadu. The sacred centerpiece is a decorated earthen or clay pot boiling over with rice, milk, and jaggery under open sky, symbolizing prosperity. Devotees wear traditional silk veshti/dhoti and silk Kanjeevaram sarees, draw intricate white and colored rice-flour Kolam patterns at entryways, and decorate with banana leaves and sugarcane stalks.",
            "visual_keywords": ["overflowing decorated earthen Pongal pot centerpiece", "kolam pattern", "kanjeevaram saree", "silk veshti", "sugarcane stalks", "marigold garland", "boiled rice pot"]
        },
        "durga": {
            "title": "Durga Puja",
            "en_summary": "Durga Puja is a major Bengali festival honoring goddess Durga. The grand centerpiece of every celebration is the towering, exquisitely sculpted idol of Goddess Durga (Durga Mata ji) featuring ten arms bearing divine weapons, riding a lion, placed prominently inside artistic pandals. Rituals include Dhunuchi naach (frenzied incense dance using smoking terracotta burners filled with coconut husk and camphor), wearing white saree with red borders (Lal Paar Saree), dhak drumming, and lotus offerings.",
            "visual_keywords": ["Goddess Durga idol (Durga Mata ji) sculpted centerpiece", "ten-armed Durga Mata shrine", "dhunuchi burner", "lal paar saree", "terracotta pandal", "dhak drum", "red border saree", "alta on hands", "lotus offerings"]
        },
        "kathakali": {
            "title": "Kathakali",
            "en_summary": "Kathakali is a traditional classical dance-drama from Kerala. Features a central performer with green facial makeup (pacca) for noble characters, elaborate tiered wooden crowns (kireetam), wide flared hoop skirts (uduthukettu), intricate facial mudras and wide expressive eyes with red-stained sclera. Performed in temple courtyards beside a large brass oil lamp (nilavilakku).",
            "visual_keywords": ["pacca green makeup performer centerpiece", "kireetam crown", "hoop skirt", "nilavilakku brass lamp", "temple courtyard", "facial mudra", "expressive eyes"]
        },
        "gudi padwa": {
            "title": "Gudi Padwa",
            "en_summary": "Gudi Padwa is the traditional New Year festival celebrated in Maharashtra. The central festive centerpiece is the Gudi flag hoisted outside homes—a bamboo stick draped with a green or yellow silk cloth topped with neem leaves, sugar candy garland (gaathi), marigolds, and an inverted silver or brass pot (kalash). Women wear traditional nine-yard Nauvari sarees with nose rings (nath) and pearl jewelry.",
            "visual_keywords": ["Gudi flag centerpiece", "nauvari saree", "inverted brass pot kalash", "gaathi sugar garland", "marigold floral wreath", "marathi nath", "rangoli design"]
        },
        "onam": {
            "title": "Onam",
            "en_summary": "Onam is the harvest festival of Kerala. Celebrated with the grand Pookkalam (intricate multi-colored fresh floral carpet arrangement centerpiece on floor), Vallam Kali (snake boat races on backwaters), Sadhya feast served on green banana leaves, and women performing Kaikottikali dance wearing off-white Kasavu sarees with gold zari borders.",
            "visual_keywords": ["grand Pookkalam floral carpet centerpiece", "kasavu saree with gold zari", "banana leaf sadhya", "snake boat", "gold jewelry", "jasmine flowers in hair"]
        }
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
        
        if not results:
            wiki_fetched_count = 0
            for term in search_terms[:2]:
                wiki_data = self.fetch_wikipedia_content(term)
                if wiki_data and wiki_data['title'] not in [r.get('title') for r in results]:
                    results.append({
                        "source": f"Wikipedia ({wiki_data['title']})",
                        "title": wiki_data['title'],
                        "text": wiki_data['summary'],
                        "full_content": wiki_data['content'],
                        "url": wiki_data['url']
                    })
                    wiki_fetched_count += 1
                    if wiki_fetched_count >= 1:
                        break

        if not results:
            translated = self.translate_query_to_english(user_query)
            results.append({
                "source": "Synthesized Regional Cultural Retrieval",
                "title": f"Cultural context for {translated}",
                "text": f"Traditional Indian cultural celebration depicting {translated} with authentic regional attire, brass accessories, ritual offerings, floral decorations, and architectural background motifs."
            })
            
        return results
