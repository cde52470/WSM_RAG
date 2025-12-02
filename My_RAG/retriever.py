from rank_bm25 import BM25Okapi
import jieba
import os
import math
from ollama import Client


def _get_ollama_client():
    """Create an Ollama client.

    在 TA 的 Docker 環境裡，gateway 是 http://ollama-gateway:11434。
    如果你本地測試有設 OLLAMA_HOST，就會用環境變數的設定。
    """
    host = os.environ.get("OLLAMA_HOST", "http://ollama-gateway:11434")
    return Client(host=host)


class BM25Retriever:
    """純 BM25 的 retriever（當作 hybrid 的第一階段 filter）。"""

    def __init__(self, chunks, language: str = "en"):
        self.chunks = chunks
        self.language = language
        self.corpus = [chunk['page_content'] for chunk in chunks]

        if language == "zh":
            self.tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]
        else:
            # 英文就用簡單的空白分詞
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
    """Hybrid retriever: BM25 pre-filter + dense embedding re-ranking.

    - 第一步：用 BM25 從所有 chunks 選出 candidate_k 個候選。
    - 第二步：用 Ollama 的 embedding 模型計算 query / 候選 chunks 的向量，
      用 cosine similarity 對候選做 re-ranking。
    - 回傳：re-ranking 後的前 top_k 個 chunks。
    """

    def __init__(self, chunks, language: str = "en", candidate_k: int = 30):
        self.chunks = chunks
        self.language = language
        self.candidate_k = candidate_k
        self.bm25_retriever = BM25Retriever(chunks, language)

        # 依語言選 embedding model（可以之後改成你想試的 model）
        if language == "zh":
            # 中文用 Qwen3 embedding
            self.embedding_model = os.environ.get("EMBED_MODEL_ZH", "qwen3-embedding:0.6b")
        else:
            # 英文用 embeddinggemma
            self.embedding_model = os.environ.get("EMBED_MODEL_EN", "embeddinggemma:300m")

        self.client = _get_ollama_client()

    def _embed_single(self, text: str):
        """計算單一句子的 embedding，兼容 dict / 物件兩種回傳型態。"""
        try:
            resp = self.client.embed(model=self.embedding_model, input=[text])
        except Exception as e:
            print(f"[HybridRetriever] Error embedding single text: {e}")
            return None

        if isinstance(resp, dict):
            embeddings = resp.get("embeddings", [])
        else:
            embeddings = getattr(resp, "embeddings", [])

        if not embeddings:
            return None

        return embeddings[0]  # 我們只丟了一句

    def _embed_batch(self, texts):
        """一批 texts 一次做 embedding。"""
        try:
            resp = self.client.embed(model=self.embedding_model, input=texts)
        except Exception as e:
            print(f"[HybridRetriever] Error embedding batch: {e}")
            return []

        if isinstance(resp, dict):
            embeddings = resp.get("embeddings", [])
        else:
            embeddings = getattr(resp, "embeddings", [])

        return embeddings

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
        if not cand_embs:
            return candidates[:top_k]

        # 4) cosine similarity re-ranking
        scored = []
        for chunk, emb in zip(candidates, cand_embs):
            sim = self._cosine_similarity(query_emb, emb)
            scored.append((chunk, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]]


def create_retriever(chunks, language):
    """工廠函式：現在回傳 hybrid BM25 + embedding 的 retriever。

    如果之後想切回純 BM25，只要改成：
        return BM25Retriever(chunks, language)
    """
    return HybridBM25EmbeddingRetriever(chunks, language)
