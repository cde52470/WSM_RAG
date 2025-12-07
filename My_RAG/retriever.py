from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import jieba
import re
from rank_bm25 import BM25Okapi
from ollama import Client
from generator import load_ollama_config

# 移除原本的 Ollama Client，因為 Rerank 不建議用生成式模型
try:
    from sentence_transformers import SentenceTransformer, CrossEncoder
except ImportError:
    SentenceTransformer = None
    CrossEncoder = None

class HybridRetriever:
    def __init__(
        self,
        chunks: List[Dict[str, Any]],
        language: str = "en",
        bm25_top_k: int = 50,      # BM25 取前 50
        embed_top_k: int = 50,     # 向量取前 50
        final_top_k: int = 5,      # 最終回傳前 5
        # rerank_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2", # 推薦的輕量級 Reranker
        rerank_model_name: str = "BAAI/bge-reranker-base", # 將預設模型換成支援中英雙語的 BAAI/bge-reranker-base
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        use_reranker: bool = True
    ):
        self.language = language
        # 1. 過濾語言
        self.chunks = [
            c for c in chunks
            if not language
            or c.get("metadata", {}).get("language") == language
            or c.get("language") == language
        ]
        self.corpus = [chunk["page_content"] for chunk in self.chunks]
        
        self.bm25_top_k = bm25_top_k
        self.embed_top_k = embed_top_k
        self.final_top_k = final_top_k
        self.use_reranker = use_reranker

        # 2. 初始化 BM25
        self.tokenized_corpus = [self._tokenize(doc) for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # 3. 初始化 Embedding Model (Bi-Encoder)
        self.dense_encoder = None
        self.chunk_embeddings = None
        if SentenceTransformer:
            self.dense_encoder = SentenceTransformer(embedding_model_name)
            # 預計算所有 chunks 的向量 (實務上這步應該在存資料庫時就做好了，不要每次 init 做)
            # 這裡使用 encode(show_progress_bar=False) 且直接轉 numpy，比迴圈快
            self.chunk_embeddings = self.dense_encoder.encode(
                self.corpus, convert_to_numpy=True, normalize_embeddings=True
            )

        # 4. 初始化 Reranker (Cross-Encoder)
        self.reranker = None
        if CrossEncoder and use_reranker:
            # 這是一個專門用來評分 (Query, Document) 相關性的模型，速度極快
            self.reranker = CrossEncoder(rerank_model_name)

        try:
            config = load_ollama_config()
            ollama_host = config.get("host", "http://ollama-gateway:11434")
        except Exception:
            ollama_host = "http://ollama-gateway:11434"
            
        self.ollama_client = Client(host=ollama_host)

    def _tokenize(self, text: str):
        if self.language == "zh":
            return list(jieba.cut(text))
        return text.split()

    def _focus_terms(self, query):
        """Extract company-like tokens and years to boost exact matches."""
        # This logic is absorbed from the 'wang' branch to improve entity recall
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9&'\\-\\.]*|\\d{4}", query)
        lower_tokens = {t.lower() for t in tokens if len(t) > 2}
        years = {t for t in lower_tokens if t.isdigit()}
        names = lower_tokens - years
        return names, years

    def _rrf_fusion(self, bm25_indices: List[int], embed_indices: List[int], k: int = 60) -> List[int]:
        """
        Reciprocal Rank Fusion (RRF):
        不依賴絕對分數，而是依賴「排名」。解決了 BM25 分數和 Cosine 分數範圍不同的問題。
        """
        rrf_score = {}

        # 處理 BM25 排名
        for rank, idx in enumerate(bm25_indices):
            if idx not in rrf_score: rrf_score[idx] = 0
            rrf_score[idx] += 1 / (k + rank + 1)

        # 處理 Vector 排名
        for rank, idx in enumerate(embed_indices):
            if idx not in rrf_score: rrf_score[idx] = 0
            rrf_score[idx] += 1 / (k + rank + 1)

        # 根據 RRF 分數排序，由高到低
        sorted_indices = sorted(rrf_score.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, score in sorted_indices]

    def _generate_hyde_doc(self, query: str) -> str:
        """
        使用 LLM 生成一個假設性的答案 (Hypothetical Document)
        """
        prompt = (
            f"Please write a passage to answer the question\n"
            f"Question: {query}\n"
            f"Passage:"
        )
        try:
            # 這裡用一個比較快的小模型生成即可，例如 llama3:8b 或 gemma
            response = self.ollama_client.generate(model = "granite4:3b", prompt=prompt)
            return response.get("response", "").strip()
        except Exception as e:
            print(f"HyDE generation failed: {e}")
            return query  # 如果生成失敗，退回使用原始 query

    def retrieve(self, query: str, top_k: int = None, use_hyde: bool = False) -> List[Dict[str, Any]]:
        final_k = top_k if top_k is not None else self.final_top_k
        
        # --- 1. 決定用於向量檢索的文字 ---
        search_text_for_vector = query
        if use_hyde and self.dense_encoder:
            # 如果開啟 HyDE，先生成假答案，用假答案去轉向量
            fake_doc = self._generate_hyde_doc(query)
            # print(f"HyDE Generated: {fake_doc[:50]}...") # debug用
            search_text_for_vector = fake_doc
        
        query_tokens = self._tokenize(query)
        
        # --- 階段 1: BM25 檢索 ---
        # get_scores 比較快，不要用 get_top_n
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # [Optimization from wang branch]
        # Light re-ranking: boost exact company/year matches to reduce entity confusion.
        names, years = self._focus_terms(query)
        if names or years:
            for i, chunk in enumerate(self.chunks):
                text_lower = chunk.get("page_content", "").lower()
                boost = 0.0
                if names:
                    boost += 0.3 * sum(1 for name in names if name in text_lower)
                if years:
                    boost += 0.15 * sum(1 for yr in years if yr in text_lower)
                if boost:
                    bm25_scores[i] += boost
        
        # 取得前 N 名的 index
        bm25_top_indices = np.argsort(bm25_scores)[::-1][:self.bm25_top_k]

        # --- 階段 2: 向量檢索 (Vector Search) ---
        embed_top_indices = []
        if self.dense_encoder and self.chunk_embeddings is not None:
            query_embedding = self.dense_encoder.encode(query, convert_to_numpy=True, normalize_embeddings=True)
            
            # 使用矩陣運算一次計算所有相似度 (Dot Product 因為已 Normalize = Cosine Similarity)
            # 這是 Numpy 的廣播機制，比 for loop 快非常多
            similarities = np.dot(self.chunk_embeddings, query_embedding)
            embed_top_indices = np.argsort(similarities)[::-1][:self.embed_top_k]

        # --- 階段 3: 融合 (Hybrid Fusion) ---
        # 使用 RRF 合併兩種結果，取前 N 個候選人進入 Rerank
        merged_indices = self._rrf_fusion(bm25_top_indices, embed_top_indices)
        
        # 這裡的候選集數量可以稍微多一點，例如取前 50 個給 Reranker
        candidate_indices = merged_indices[:50] 
        candidate_docs = [self.corpus[i] for i in candidate_indices]

        # --- 階段 4: 重排序 (Re-ranking) ---
        if self.reranker:
            # Cross-Encoder 接受 list of pairs: [(query, doc1), (query, doc2), ...]
            pairs = [[query, doc] for doc in candidate_docs]
            rerank_scores = self.reranker.predict(pairs)
            
            # 結合 index 和分數
            results_with_scores = list(zip(candidate_indices, rerank_scores))
            # 根據 rerank 分數重新排序
            results_with_scores.sort(key=lambda x: x[1], reverse=True)
            
            final_indices = [idx for idx, score in results_with_scores[:self.final_top_k]]
        else:
            # 如果沒有 Reranker，直接回傳 RRF 的結果
            final_indices = candidate_indices[:self.final_top_k]

        return [self.chunks[idx] for idx in final_indices]

# 使用範例
def create_retriever(chunks, language="en"):
    return HybridRetriever(chunks, language=language)
