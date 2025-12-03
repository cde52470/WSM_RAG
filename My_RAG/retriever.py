from rank_bm25 import BM25Okapi
import jieba
import os
import math
from ollama import Client


class BM25Retriever:
    """A simple BM25 retriever used as the first-stage filter."""

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
        """Returns the top_k chunks with the highest BM25 scores."""
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
        self.client = client

        if language == "zh":
            self.embedding_model = os.environ.get("EMBED_MODEL_ZH", "qwen3-embedding:0.6b")
        else:
            self.embedding_model = os.environ.get("EMBED_MODEL_EN", "embeddinggemma:300m")

    def _embed_single(self, text: str):
        """Computes the embedding for a single text."""
        if not self.client:
            return None
        try:
            resp = self.client.embeddings(model=self.embedding_model, prompt=text)
            return resp.get("embedding")
        except Exception as e:
            print(f"[HybridRetriever] Error embedding single text: {e}")
            return None

    def _embed_batch(self, texts: list[str]):
        """Computes embeddings for a batch of texts."""
        if not self.client:
            return []
        try:
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
        
        vec1 = list(vec1)
        vec2 = list(vec2)

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

    def retrieve(self, query: str, top_k: int = 5):
        # 1) Use BM25 to get candidate_k candidates
        candidates = self.bm25_retriever.retrieve(query, top_k=self.candidate_k)
        if not candidates:
            return []

        # 2) Compute query embedding
        query_emb = self._embed_single(query)
        if not query_emb:
            return candidates[:top_k]

        # 3) Compute embeddings for candidate chunks
        cand_texts = [c["page_content"] for c in candidates]
        cand_embs = self._embed_batch(cand_texts)
        if not cand_embs or len(cand_embs) != len(candidates):
            return candidates[:top_k]

        # 4) Re-rank using cosine similarity
        scored_candidates = []
        for chunk, emb in zip(candidates, cand_embs):
            if emb:
                sim = self._cosine_similarity(query_emb, emb)
                scored_candidates.append((chunk, sim))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        return [chunk for chunk, score in scored_candidates[:top_k]]


def create_retriever(chunks, language, client):
    """Factory function for creating a hybrid retriever."""
    return HybridBM25EmbeddingRetriever(chunks, language, client=client)
