import os
import re
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
import ollama
# ==========================================
# 1. Configuration: 中英文調整
# ==========================================
class RAGConfig:
    SETTINGS = {
        "zh": {
            "vector_model": "qwen2.5:0.5b",       # 中文模型
            "bm25_tokenizer": "jieba",            # 中文斷詞
            "weights": {"bm25": 0.4, "vec": 0.6}, # 中文語意較依賴向量
        },
        "en": {
            "vector_model": "nomic-embed-text",   # 英文模型
            "bm25_tokenizer": "space",
            "weights": {"bm25": 0.5, "vec": 0.5}, # 英文關鍵字通常很準
        }
    }

    @classmethod
    def get(cls, lang, key):
        cfg = cls.SETTINGS.get(lang, cls.SETTINGS["en"])
        return cfg.get(key)

# ==========================================
# 2. 檢索模型 (Retrieval Models)
# ==========================================

class SparseRetriever:
    """
    模型 1: BM25
    """
    def __init__(self, chunks, language):
        self.corpus = [chunk["page_content"] for chunk in chunks]
        self.language = language
        self.tokenizer_type = RAGConfig.get(language, "bm25_tokenizer")
        
        # 建立索引
        if self.tokenizer_type == "jieba":
            self.tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]
        else:
            self.tokenized_corpus = [doc.lower().split(" ") for doc in self.corpus]
            
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def search(self, query, top_k=50):
        # 處理 Query
        if self.tokenizer_type == "jieba":
            tokenized_query = list(jieba.cut(query))
        else:
            tokenized_query = query.lower().split(" ")

        scores = self.bm25.get_scores(tokenized_query)
        
        # 格式化輸出: List of (index, score)
        results = []
        for idx, score in enumerate(scores):
            # 過濾掉分數極低或為 0 的結果
            if score > 1e-5: 
                results.append((idx, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

class DenseRetriever:
    """
    模型 2: Vector Search (符合作業建議的第二種模型)
    """
    def __init__(self, chunks, language):
        self.chunks = chunks
        self.language = language
        self.model = RAGConfig.get(language, "vector_model")
        self.embeddings = []
        
        # 預先計算所有文件的向量 (Indexing)
        self._build_index()

    def _get_embedding(self, text):
        if not ollama: return np.zeros(768)
        try:
            # 使用 Ollama API
            resp = ollama.embeddings(model=self.model, prompt=text)
            return np.array(resp["embedding"])
        except Exception as e:
            print(f"[Error] Embedding failed: {e}")
            return np.zeros(768)

    def _build_index(self):
        print(f"[{self.language}] Vector Indexing start...")
        for chunk in self.chunks:
            vec = self._get_embedding(chunk["page_content"])
            self.embeddings.append(vec)
        self.embeddings = np.array(self.embeddings)

    def search(self, query, top_k=50):
        query_vec = self._get_embedding(query)
        if query_vec is None or len(self.embeddings) == 0:
            return []

        # Cosine Similarity 計算
        # Formula: (A . B) / (|A| * |B|)
        q_norm = np.linalg.norm(query_vec)
        d_norms = np.linalg.norm(self.embeddings, axis=1)
        
        # 避免除以 0
        d_norms[d_norms == 0] = 1e-10
        if q_norm == 0: q_norm = 1e-10

        dot_products = np.dot(self.embeddings, query_vec)
        similarities = dot_products / (q_norm * d_norms)

        results = [(i, float(score)) for i, score in enumerate(similarities)]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

# ==========================================
# 3. 進階重排序 (Advanced Re-ranking)
#    對應作業：Advanced Tasks / Classifier
# ==========================================

class RelevanceClassifier:
    """
    這是一個 Classifier 的框架。
    目前實作：基於規則 (Heuristic) 的分類器。
    進階實作：你可以訓練一個 Logistic Regression 或使用 Cross-Encoder 來取代這裡的邏輯。
    """
    def predict_score(self, query, chunk_content, original_score):
        boost = 0.0
        
        # Feature 1: 實體匹配 (Entity Matching)
        # 這是作業提到的 "Advanced Task" 的一種簡單實現
        query_terms = set(re.findall(r"\w+", query.lower()))
        chunk_terms = set(re.findall(r"\w+", chunk_content.lower()))
        
        # 找出年份 (如 2023, 112學年) - 這在學校作業中通常是考點
        years = re.findall(r"\d{4}", query)
        for year in years:
            if year in chunk_content:
                boost += 0.2  # 年份匹配給予高權重
        
        # Feature 2: 關鍵詞覆蓋率 (Term Overlap)
        overlap = len(query_terms & chunk_terms)
        boost += overlap * 0.05

        return original_score + boost

# ==========================================
# 4. 主流程 (Main Pipeline)
#    對應作業：Fusion
# ==========================================

class EnsembleRetriever:
    def __init__(self, chunks, language="en"):
        self.chunks = chunks
        self.language = language
        self.weights = RAGConfig.get(language, "weights")
        
        # 初始化模型
        self.bm25_retriever = SparseRetriever(chunks, language)
        self.vector_retriever = DenseRetriever(chunks, language)
        self.classifier = RelevanceClassifier()

    def _normalize(self, results):
        """
        Normalization: 將分數映射到 [0, 1]
        BM25: 0 ~ inf -> 0 ~ 1
        Vector: -1 ~ 1 -> 0 ~ 1 (雖然通常是 0~1)
        """
        if not results: return {}
        
        scores = [r[1] for r in results]
        min_s, max_s = min(scores), max(scores)
        
        norm_map = {}
        for idx, score in results:
            if max_s - min_s == 0:
                norm_map[idx] = 1.0 if max_s > 0 else 0.0
            else:
                norm_map[idx] = (score - min_s) / (max_s - min_s)
        return norm_map

    def retrieve(self, query, top_k=10):
        # 1. 雙路召回 (Retrieval)
        # 為了融合效果，這裡取較多的候選集 (top_k * 3)
        candidates_k = top_k * 3
        bm25_res = self.bm25_retriever.search(query, top_k=candidates_k)
        vec_res = self.vector_retriever.search(query, top_k=candidates_k)

        # 2. 分數歸一化 (Normalization)
        bm25_norm = self._normalize(bm25_res)
        vec_norm = self._normalize(vec_res)

        # 3. 加權融合 (Weighted Sum Fusion)
        all_indices = set(bm25_norm.keys()) | set(vec_norm.keys())
        merged_results = []
        
        alpha = self.weights["vec"]
        beta = self.weights["bm25"]

        for idx in all_indices:
            s_bm25 = bm25_norm.get(idx, 0.0)
            s_vec = vec_norm.get(idx, 0.0)
            
            # 融合公式
            fusion_score = (beta * s_bm25) + (alpha * s_vec)
            
            merged_results.append({
                "index": idx,
                "score": fusion_score,
                "chunk": self.chunks[idx]
            })

        # 4. Re-ranking (使用 Classifier / Heuristic)
        # 這是作業的 Advanced Task 部分
        for item in merged_results:
            new_score = self.classifier.predict_score(
                query, 
                item["chunk"]["page_content"], 
                item["score"]
            )
            item["score"] = new_score

        # 5. 最終排序
        merged_results.sort(key=lambda x: x["score"], reverse=True)
        
        # 回傳 Top-K 的原始 chunk 內容
        return [item["chunk"] for item in merged_results[:top_k]]

def create_retriever(chunks, language):
    return EnsembleRetriever(chunks, language)