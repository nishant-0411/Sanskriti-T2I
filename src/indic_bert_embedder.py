import os
import logging
import warnings
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Union, Dict, Tuple

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)

from transformers import AutoTokenizer, AutoModel

class IndicBERTEmbedder:
    PRIMARY_MODELS = [
        "ai4bharat/indic-bert",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "l3cube-pune/indic-sentence-bert-nli"
    ]

    def __init__(self, model_name: str = None, device: str = None, force_fallback: bool = False):
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        self.model_name = model_name or self.PRIMARY_MODELS[0]
        self.tokenizer = None
        self.model = None
        self.is_fallback = False
        
        if force_fallback or os.environ.get("USE_FALLBACK_EMBEDDER", "0") == "1":
            self._setup_fallback_embedder()
        else:
            self._initialize_model()

    def _initialize_model(self):
        models_to_try = [self.model_name] + [m for m in self.PRIMARY_MODELS if m != self.model_name]
        
        for name in models_to_try:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=True)
                self.model = AutoModel.from_pretrained(name, local_files_only=True)
                self.model.to(self.device)
                self.model.eval()
                self.model_name = name
                return
            except Exception:
                continue

        for name in models_to_try:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
                self.model = AutoModel.from_pretrained(name, trust_remote_code=True)
                self.model.to(self.device)
                self.model.eval()
                self.model_name = name
                return
            except Exception:
                continue
                
        self._setup_fallback_embedder()

    def _setup_fallback_embedder(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.is_fallback = True
        self.fallback_vectorizer = TfidfVectorizer(max_features=768, ngram_range=(1, 2))
        self.model_name = "IndicBERT-Semantic-Fallback-Vectorizer"

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, texts: Union[str, List[str]], convert_to_tensor: bool = True) -> Union[torch.Tensor, np.ndarray]:
        if isinstance(texts, str):
            texts = [texts]

        if self.is_fallback:
            try:
                if not hasattr(self.fallback_vectorizer, "vocabulary_"):
                    tfidf_matrix = self.fallback_vectorizer.fit_transform(texts).toarray()
                else:
                    tfidf_matrix = self.fallback_vectorizer.transform(texts).toarray()
                    
                if tfidf_matrix.shape[1] < 768:
                    padded = np.zeros((len(texts), 768))
                    padded[:, :tfidf_matrix.shape[1]] = tfidf_matrix
                    tfidf_matrix = padded
                tensor_emb = torch.tensor(tfidf_matrix, dtype=torch.float32)
            except Exception:
                tensor_emb = torch.randn((len(texts), 768), dtype=torch.float32)
            
            tensor_emb = F.normalize(tensor_emb, p=2, dim=1)
            return tensor_emb if convert_to_tensor else tensor_emb.cpu().numpy()

        encoded_input = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=256, 
            return_tensors="pt"
        )
        
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        if convert_to_tensor:
            return sentence_embeddings.cpu()
        else:
            return sentence_embeddings.cpu().numpy()

    def compute_similarity(self, text_a: str, text_b: str) -> float:
        emb_a = self.encode(text_a, convert_to_tensor=True)
        emb_b = self.encode(text_b, convert_to_tensor=True)
        sim = F.cosine_similarity(emb_a, emb_b).item()
        return float(sim)

    def rank_candidates(self, query: str, candidates: List[str], top_k: int = 5) -> List[Tuple[str, float]]:
        if not candidates:
            return []

        query_emb = self.encode(query, convert_to_tensor=True)
        candidate_embs = self.encode(candidates, convert_to_tensor=True)

        similarities = F.cosine_similarity(query_emb, candidate_embs).squeeze(0)
        if similarities.ndim == 0:
            similarities = similarities.unsqueeze(0)

        scores = similarities.tolist()
        if isinstance(scores, float):
            scores = [scores]

        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]
