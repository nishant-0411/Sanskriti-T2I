import re
from typing import List, Dict, Any, Tuple
import torch
import torch.nn.functional as F
from src.indic_bert_embedder import IndicBERTEmbedder

class CulturalContextFilter:
    VISUAL_CATEGORIES = {
        "attire_jewelry": [
            "traditional clothing saree dhoti veshti turban mukut jewelry alta solah shringar gold zari border silk cotton garment ornament necklace bangles earings crown dress attire worn"
        ],
        "rituals_actions": [
            "ritual action pose prayer offering worship dance mudra holding arghya folded hands namaskar festive ritual ceremony sacred rite dancing celebration kneeling standing"
        ],
        "architecture_setting": [
            "architecture mandap ghat riverbank pandal temple courtyard arch pillar carved wood stone shrine doorway veranda background setting structure design building"
        ],
        "objects_artifacts": [
            "brass lamp diya matka soop thali clay pot incense burner dhunuchi coconut banana leaf flowers garland kalash vessel earthen pot tray drum"
        ],
        "environment_atmosphere": [
            "atmosphere lighting morning sunlight golden hour dusk candle flame incense smoke mist outdoor daylight warm glow sky reflection river sunrise sunset soft lighting"
        ],
        "colors_textures": [
            "vermilion red turmeric yellow saffron gold royal peacock blue terracotta crimson emerald silk weave pattern vibrant color palette bright gold white"
        ],
        "regional_characteristics": [
            "regional motifs traditional ethnic folk authentic local heritage cultural style regional identity bengali kerala rajasthani tamil maharashtrian punjabi heritage"
        ]
    }

    NON_VISUAL_KEYWORDS = [
        "population", "census", "kilometer", "sq mi", "government", "minister", "election",
        "district", "municipality", "gdp", "economy", "railway", "highway", "transport",
        "century", "bce", "ce", "etymology", "derived from", "wikipedia", "article", "references"
    ]

    def __init__(self, embedder: IndicBERTEmbedder, max_tokens: int = 150):
        self.embedder = embedder
        self.max_tokens = max_tokens
        self.category_embeddings = {}
        self._precompute_category_embeddings()

    def _precompute_category_embeddings(self):
        """Precomputes IndicBERT dense embeddings for visual category prototypes."""
        for cat, desc_list in self.VISUAL_CATEGORIES.items():
            emb = self.embedder.encode(desc_list[0], convert_to_tensor=True)
            self.category_embeddings[cat] = emb

    def sentence_tokenize(self, text: str) -> List[str]:
        """Splits retrieved context text into sentences and short descriptive clauses."""
        text = re.sub(r'\[\d+\]', '', text) # Remove citations like [1]
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = []
        for s in raw_sentences:
            s_clean = s.strip()
            if len(s_clean) > 15:
                clauses = re.split(r';\s*', s_clean)
                sentences.extend([c.strip() for c in clauses if len(c.strip()) > 15])
        return sentences

    def is_visual_content(self, text: str) -> bool:
        text_lower = text.lower()
        if any(nv in text_lower for nv in self.NON_VISUAL_KEYWORDS):
            return False
        return True

    def filter_and_categorize(self, user_query: str, context_blocks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        all_sentences = []
        for block in context_blocks:
            text = block.get("text", "")
            full_text = block.get("full_content", "")
            combined = f"{text}. {full_text}" if full_text else text
            sentences = self.sentence_tokenize(combined)
            all_sentences.extend(sentences)

            if "visual_keywords" in block:
                for kw in block["visual_keywords"]:
                    all_sentences.append(f"Visual detail: {kw}")

        unique_sentences = list(dict.fromkeys(all_sentences))
        visual_sentences = [s for s in unique_sentences if self.is_visual_content(s)]
        
        if not visual_sentences:
            return {cat: [] for cat in self.VISUAL_CATEGORIES}

        sentence_embs = self.embedder.encode(visual_sentences, convert_to_tensor=True)
        query_emb = self.embedder.encode(user_query, convert_to_tensor=True)

        categorized_details: Dict[str, List[Tuple[str, float]]] = {cat: [] for cat in self.VISUAL_CATEGORIES}

        for idx, sentence in enumerate(visual_sentences):
            s_emb = sentence_embs[idx:idx+1]
            
            q_sim = F.cosine_similarity(query_emb, s_emb).item()

            best_cat = None
            best_cat_sim = -1.0

            for cat_name, cat_emb in self.category_embeddings.items():
                cat_sim = F.cosine_similarity(cat_emb, s_emb).item()
                if cat_sim > best_cat_sim:
                    best_cat_sim = cat_sim
                    best_cat = cat_name

            combined_score = (0.4 * q_sim) + (0.6 * best_cat_sim)

            if best_cat and combined_score > 0.25:
                categorized_details[best_cat].append((sentence, combined_score))

        final_categorized: Dict[str, List[str]] = {}
        total_words = 0

        for cat_name, items in categorized_details.items():
            items.sort(key=lambda x: x[1], reverse=True)
            selected = []
            for text_str, score in items[:3]:
                word_count = len(text_str.split())
                if total_words + word_count <= self.max_tokens:
                    selected.append(text_str)
                    total_words += word_count
                else:
                    break
            final_categorized[cat_name] = selected

        return final_categorized

    def compress_to_visual_tokens(self, categorized: Dict[str, List[str]]) -> str:
        """
        Compresses categorized visual details into a concise, token-efficient summary.
        """
        token_parts = []
        for cat, details in categorized.items():
            if details:
                label = cat.replace("_", " ").title()
                clean_details = "; ".join(details)
                token_parts.append(f"[{label}]: {clean_details}")
        return " | ".join(token_parts)
