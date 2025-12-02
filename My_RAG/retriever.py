from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import jieba
from rank_bm25 import BM25Okapi

from generator import load_ollama_config

try:
    from ollama import Client
except ImportError:
    Client = None
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


def _safe_cosine(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class HybridRetriever:
    """
    Hybrid retrieval = BM25 + Embedding similarity, followed by optional LLM rerank.
    BM25 provides exact token recall, embeddings broaden semantics, reranker sharpens the top list.
    """

    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        language: str = "en",
        alpha: float = 0.5,
        bm25_top_n: int = 50,
        embed_top_n: int = 50,
        rerank_top_n: int = 50,
        rerank_model: Optional[str] = None,
        embedding_model: Optional[str] = None,
        dense_model: Optional[str] = None,
        client: Optional[Any] = None,
    ):
        self.language = language
        self.chunks = [
            c
            for c in chunks
            if not language
            or c.get("metadata", {}).get("language") == language
            or c.get("language") == language
        ]
        self.alpha = alpha
        self.bm25_top_n = bm25_top_n
        self.embed_top_n = embed_top_n
        self.rerank_top_n = rerank_top_n

        self.corpus = [chunk["page_content"] for chunk in self.chunks]
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        self.ollama_client = client # Use passed client
        self.embedding_model = embedding_model
        self.rerank_model = rerank_model
        self.dense_model_name = dense_model
        self.dense_encoder: Optional[SentenceTransformer] = None
        self.chunk_embeddings: List[Optional[np.ndarray]] = [None] * len(self.chunks)

        if self.ollama_client:
             try:
                config = load_ollama_config()
                # host = config.get("host", "http://localhost:11434") # Host is handled by client
                # self.ollama_client = Client(host=host) # Removed
                self.embedding_model = embedding_model or config.get("embedding_model") or config.get("model")
                self.rerank_model = rerank_model or config.get("rerank_model") or config.get("model")
             except Exception:
                # self.ollama_client = None # Don't set to None if passed, just maybe warn?
                pass

        if SentenceTransformer is not None:
            try:
                config = load_ollama_config()
                self.dense_model_name = dense_model or config.get("dense_model") or "sentence-transformers/all-MiniLM-L6-v2"
                self.dense_encoder = SentenceTransformer(self.dense_model_name)
            except Exception:
                self.dense_encoder = None

        if self.ollama_client and self.embedding_model:
            self._precompute_embeddings()
        elif self.dense_encoder:
            self._precompute_embeddings()

    def _tokenize(self, text: str):
        if self.language == "zh":
            return list(jieba.cut(text))
        return text.split()

    def _embed_text(self, text: str) -> Optional[np.ndarray]:
        if self.dense_encoder is not None:
            try:
                return np.asarray(self.dense_encoder.encode(text, normalize_embeddings=True))
            except Exception:
                return None
        if self.ollama_client and self.embedding_model:
            try:
                resp = self.ollama_client.embeddings(model=self.embedding_model, prompt=text)
                embedding = resp.get("embedding") or (resp.get("data") or [{}])[0].get("embedding")
                return np.array(embedding, dtype=float) if embedding else None
            except Exception:
                return None
        return None

    def _precompute_embeddings(self):
        for idx, text in enumerate(self.corpus):
            self.chunk_embeddings[idx] = self._embed_text(text)

    def _hybrid_scores(self, query_tokens: List[str], query_embedding: Optional[np.ndarray]) -> List[Tuple[int, float]]:
        bm25_scores = self.bm25.get_scores(query_tokens)

        bm25_top_idx = np.argsort(bm25_scores)[::-1][: self.bm25_top_n]

        embed_scores = np.zeros(len(self.chunks))
        if query_embedding is not None:
            for idx, emb in enumerate(self.chunk_embeddings):
                if emb is not None:
                    embed_scores[idx] = _safe_cosine(query_embedding, emb)
        embed_top_idx = np.argsort(embed_scores)[::-1][: self.embed_top_n]

        candidate_indices = set(bm25_top_idx.tolist() + embed_top_idx.tolist())

        hybrid = []
        for idx in candidate_indices:
            bm25_score = bm25_scores[idx]
            embed_score = embed_scores[idx] if query_embedding is not None else 0.0
            combined = self.alpha * bm25_score + (1 - self.alpha) * embed_score
            hybrid.append((idx, combined))

        hybrid.sort(key=lambda x: x[1], reverse=True)
        return hybrid

    def _rerank(self, query: str, candidates: List[Tuple[int, float]]) -> List[Tuple[int, float]]:
        if not self.ollama_client or not self.rerank_model:
            return candidates

        reranked: List[Tuple[int, float]] = []
        for idx, _ in candidates[: self.rerank_top_n]:
            doc = self.chunks[idx]["page_content"]
            prompt = (
                "Given the query and document, provide a single relevance score between 0 and 1. "
                "Respond with only the number.\n"
                f"Query: {query}\n"
                f"Document: {doc}\n"
                "Relevance score:"
            )
            try:
                resp = self.ollama_client.generate(model=self.rerank_model, prompt=prompt)
                text = resp.get("response", "").strip()
                score = float(text.split()[0])
            except Exception:
                score = 0.0
            reranked.append((idx, score))

        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    def retrieve(self, query: str, top_k: int = 5):
        query_tokens = self._tokenize(query)
        query_embedding = self._embed_text(query) if self.ollama_client and self.embedding_model else None

        hybrid_candidates = self._hybrid_scores(query_tokens, query_embedding)
        reranked = self._rerank(query, hybrid_candidates)

        final = reranked[:top_k]
        return [self.chunks[idx] for idx, _ in final]


def create_retriever(
    chunks: List[Dict[str, Any]],
    language: str,
    alpha: float = 0.5,
    bm25_top_n: int = 50,
    embed_top_n: int = 50,
    rerank_top_n: int = 50,
    dense_model: Optional[str] = None,
    client: Optional[Any] = None,
):
    """Creates a hybrid retriever that mixes BM25, embeddings, and an optional reranker."""
    return HybridRetriever(
        chunks,
        language,
        alpha=alpha,
        bm25_top_n=bm25_top_n,
        embed_top_n=embed_top_n,
        rerank_top_n=rerank_top_n,
        dense_model=dense_model,
        client=client,
    )
