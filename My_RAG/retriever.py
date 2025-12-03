from rank_bm25 import BM25Okapi
import jieba
import os
import math
from ollama import Client


class BM25Retriever:
    """純 BM25 的 retriever（當作 hybrid 的第一階段 filter）。"""

    def __init__(self, chunks, language: str = "en"):
        self.chunks = chunks
        self.language = language
        self.corpus = [chunk['page_content'] for chunk in chunks]

        if language == "zh":
            self.tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]
        else:
            self.tokenized_corpus = [doc.split() for doc in self.corpus]

        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _tokenize_query(self, query: str):
        if self.language == "zh":
            return list(jieba.cut(query))
        else:
            return query.split()

    def retrieve(self, query: str, top_k: int = 5):
        """回傳 BM25 分數最高的 top_k 個 chunks。"""
        tokenized_query = self._tokenize_query(query)
        top_chunks = self.bm25.get_top_n(tokenized_query, self.chunks, n=top_k)
        return top_chunks


class HybridBM25EmbeddingRetriever:
    """Hybrid retriever: BM25 pre-filter + dense embedding re-ranking."""

    def __init__(self, chunks, language: str = "en", client: Client = None, candidate_k: int = 30):
        self.chunks = chunks
        self.language = language
        self.candidate_k = candidate_k
        self.bm25_retriever = BM25Retriever(chunks, language)
        self.client = client # Use passed client

        if language == "zh":
            self.embedding_model = os.environ.get("EMBED_MODEL_ZH", "qwen3-embedding:0.6b")
        else:
            self.embedding_model = os.environ.get("EMBED_MODEL_EN", "nomic-embed-text")

    def _embed_single(self, text: str):
        """計算單一句子的 embedding。"""
        if not self.client:
            return None
        try:
            resp = self.client.embeddings(model=self.embedding_model, prompt=text)
            return resp.get("embedding")
        except Exception as e:
            print(f"[HybridRetriever] Error embedding single text: {e}")
            return None

    def _embed_batch(self, texts: list[str]):
        """一批 texts 一次做 embedding。"""
        if not self.client:
            return []
        try:
            # Note: The ollama-python library's embed function is not ideal for batching as it sends one by one.
            # A more optimized client might send them in parallel.
            embeddings = []
            for text in texts:
                resp = self.client.embeddings(model=self.embedding_model, prompt=text)
                embeddings.append(resp.get("embedding"))
            return embeddings
        except Exception as e:
            print(f"[HybridRetriever] Error embedding batch: {e}")
            return []

    @staticmethod
    def _cosine_similarity(vec1, vec2):
        if not vec1 or not vec2:
            return 0.0
        dot = 0.0
        norm1 = 0.0
        norm2 = 0.0
        for a, b in zip(vec1, vec2):
            dot += a * b
            norm1 += a * a
            norm2 += b * b
        if norm1 <= 0.0 or norm2 <= 0.0:
            return 0.0
        return dot / (math.sqrt(norm1) * math.sqrt(norm2))

    def retrieve(self, query: str, top_k: int = 5):
        # 1) 先用 BM25 取 candidate_k 個候選
        candidates = self.bm25_retriever.retrieve(query, top_k=self.candidate_k)
        if not candidates:
            return []

        # 2) 計算 query embedding
        query_emb = self._embed_single(query)
        if query_emb is None:
            # 如果 embedding 掛掉，就退回純 BM25
            return candidates[:top_k]

        # 3) 候選 chunks embedding（一次 batch）
        cand_texts = [c["page_content"] for c in candidates]
        cand_embs = self._embed_batch(cand_texts)
        if not cand_embs or len(cand_embs) != len(candidates):
            return candidates[:top_k]

        # 4) cosine similarity re-ranking
        scored = []
        for chunk, emb in zip(candidates, cand_embs):
            if emb:
                sim = self._cosine_similarity(query_emb, emb)
                scored.append((chunk, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k}]


def create_retriever(chunks, language, client):
    """工廠函式：回傳 hybrid BM25 + embedding 的 retriever。"""
    return HybridBM25EmbeddingRetriever(chunks, language, client=client)
