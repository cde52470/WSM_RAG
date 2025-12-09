import os
import re
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
import ollama
# ==========================================
# 1. Dense DenseRetriever Configuration: 中英文調整
# ==========================================
class RAGConfig:
    SETTINGS = {
        "zh": {
            "vector_model": "qwen2.5:0.5b",       # 中文模型
            "bm25_tokenizer": "jieba",            
            "weights": {"bm25": 0.4, "vec": 0.6}, 
        },
        "en": {
            "vector_model": "nomic-embed-text",   # 英文模型
            "bm25_tokenizer": "space",
            "weights": {"bm25": 0.5, "vec": 0.5}, 
        }
    }

    #避免傳入未知語言（傳入例外語言視爲英文）
    @classmethod
    def get(cls, lang, key):
        cfg = cls.SETTINGS.get(lang, cls.SETTINGS["en"])
        return cfg.get(key)

# ==========================================
# 2. 檢索模型 (Retrieval Models)
# ==========================================

class SparseRetriever:
    """
    BM25
    """
    def __init__(self, chunks, language):
        #"page_content"抓出來
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
        
        # 程式剛開始跑，先花時間把所有文件轉換成向量存起來 (Indexing)
        self._build_index()

    def _get_embedding(self, text):
        #檢查Ollama
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
        #將矩陣轉換成array
        self.embeddings = np.array(self.embeddings)

    def search(self, query, top_k=50):
        query_vec = self._get_embedding(query)

        #檢查是否轉換失敗或資料庫是空的
        if query_vec is None or len(self.embeddings) == 0:
            return []

        # Cosine Similarity 計算
        #query向量的長度（｜A｜）
        q_norm = np.linalg.norm(query_vec)
        #所有文章向量的長度（｜B｜）
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
# 3. Re-ranking
# ==========================================

class RelevanceClassifier:
    """
    這是一個 Classifier 的框架。
    目前實作：基於規則 (Heuristic) 的分類器。
    進階實作：你可以訓練一個 Logistic Regression 或使用 Cross-Encoder 來取代這裡的邏輯。
    """
    def __init__(self, language= "en"):
        self.language
        pass
    def predict_score(self, query, chunk_content, original_score):
        boost = 0.0
        
        # Feature 1: 實體匹配 (Entity Matching)
        # 這是作業提到的 "Advanced Task" 的一種簡單實現
        query_terms = set(re.findall(r"\w+", query.lower()))
        chunk_terms = set(re.findall(r"\w+", chunk_content.lower()))
        
        # 找出年份 (如 2023, 112學年)
        years = re.findall(r"\d{4}", query)
        for year in years:
            if year in chunk_content:
                boost += 0.2  # 年份匹配給予高權重
        
        # Feature 2: 關鍵詞覆蓋率 (Term Overlap)
        overlap = len(query_terms & chunk_terms)
        boost += overlap * 0.05

        return original_score + boost
    
    def predict_score_with_llm(self, query, chunk_content, original_score):
        """
        使用 Granite 模型來進行重排序評分
        """
        if not ollama: 
            return original_score # 如果沒有 ollama，直接回傳原分數

        # 1. 建構 Prompt (這是 EDA 知識萃取的部分)
        # 我們把「年份」等特徵直接寫在 Prompt 裡提醒 LLM 注意
        prompt = f"""
        Task: You are a relevance judge. 
        Query: {query}
        Document: {chunk_content}
        
        Instruction: 
        1. Analyze if the Document directly answers the Query.
        2. Pay special attention to EXACT MATCHES of years (e.g., 2023, 2024) and entity names.
        3. Assign a relevance score from 0.0 to 1.0.
        4. Output ONLY the number, nothing else.
        
        Score:
        """

        try:
            # 2. 呼叫 LLM (使用跟 Generation 一樣的模型 granite4:3b)
            # 注意：這裡的 model 名稱要跟作業要求一致
            response = ollama.generate(model='granite4:3b', prompt=prompt)
            output = response['response'].strip()
            
            # 3. 解析分數 (從字串抓出數字)
            # 找小數點或整數
            match = re.search(r"0\.\d+|1\.0|\d+", output)
            if match:
                llm_score = float(match.group())
                # 確保分數在 0~1 之間
                llm_score = min(max(llm_score, 0.0), 1.0)
                
                # 4. 融合策略 (Fusion Strategy)
                # 你可以選擇完全相信 LLM，或是跟原本的分數加權
                # 建議：原本分數佔 30%，LLM 佔 70%
                final_score = (original_score * 0.3) + (llm_score * 0.7)
                return final_score
            else:
                return original_score # 解析失敗，維持原判

        except Exception as e:
            print(f"[Rerank Error] {e}")
            return original_score

    # 為了相容原本的介面，我們可以保留舊函式，或在 retrieve 裡改呼叫上面的
    def predict_score(self, query, chunk_content, original_score):
        # 這裡決定你要用「規則」還是「LLM」
        # 建議：為了速度，只對前 10 名用 LLM，後面的用規則
        return self.predict_score_with_llm(query, chunk_content, original_score)

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
        Vector: -1 ~ 1 -> 0 ~ 1
        """
        if not results: return {}
        
        scores = [r[1] for r in results]
        min_s, max_s = min(scores), max(scores)
        
        norm_map = {}
        for idx, score in results:
            if max_s - min_s == 0:
                #避免除0
                norm_map[idx] = 1.0 if max_s > 0 else 0.0
            else:
                norm_map[idx] = (score - min_s) / (max_s - min_s)
        return norm_map

    def retrieve(self, query, top_k=20):
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
            #如果另一方沒入選，是另一方得0分
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
        
        # 6.【策略】只對前 10 名進行 LLM Reranking
        top_n_rerank = 10 
        
        final_results = []
        
        for i, item in enumerate(merged_results):
            if i < top_n_rerank:
                # 前 10 名：使用 LLM 深度檢查
                new_score = self.classifier.predict_score_with_llm(
                    query, 
                    item["chunk"]["page_content"], 
                    item["score"]
                )
                item["score"] = new_score
            else:
                # 10 名以後：維持原分數 (或是只用簡單規則加分)
                # 這樣就不會浪費時間算那些根本不會贏的文章
                pass
            
            final_results.append(item)

        # 5. 最終排序
        final_results.sort(key=lambda x: x["score"], reverse=True)
        
        # 回傳 Top-K 的原始 chunk 內容
        return [item["chunk"] for item in final_results[:top_k]]

def create_retriever(chunks, language):
    return EnsembleRetriever(chunks, language)