from rank_bm25 import BM25Okapi
import jieba
import os
import math
import numpy as np
import re


class BM25Retriever:
    def __init__(self, chunks, language="en"):
        self.chunks = chunks
        self.language = language
        #建立語料庫
        self.corpus = [chunk["page_content"] for chunk in chunks]

        if language == "zh":
            self._tokenizer = jieba.Tokenizer()
            self.tokenized_corpus = [
                list(self._tokenizer.cut(doc)) for doc in self.corpus
            ]
        else:
            self._tokenizer = None
            self.tokenized_corpus = [doc.split(" ") for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _focus_terms(self, query):
        """Extract company-like tokens and years to boost exact matches."""
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9&'\\-\\.]*|\\d{4}", query)
        lower_tokens = {t.lower() for t in tokens if len(t) > 2}
        years = {t for t in lower_tokens if t.isdigit()}
        names = lower_tokens - years
        return names, years

    def retrieve_candidates(self, query, top_k=50):
        if self.language == "zh":
            tokenized_query = list(self._tokenizer.cut(query))
        else:
            tokenized_query = query.split(" ")

        #取得分數
        scores = self.bm25.get_scores(tokenized_query)
        names, years = self._focus_terms(query)

        # Light re-ranking: boost exact company/year matches to reduce entity confusion.
        for i, chunk in enumerate(self.chunks):
            text_lower = chunk["page_content"].lower()
            boost = 0.0
            if names:
                boost += 0.3 * sum(1 for name in names if name in text_lower)
            if years:
                boost += 0.15 * sum(1 for yr in years if yr in text_lower)
            if boost:
                scores[i] += boost

        #取出分數最高的TOP-k個索引
        top_k = min(top_k, len(scores))
        # 使用 numpy 高效排序
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        #按分數高低排列
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        #回傳完整的 chunk 物件列表
        return [self.chunks[i] for i in top_indices]

#Hybrid Retriever (結合 Sparse + Dense)
class HybridRetriever:
    """
    1. 先用 BM25 快速篩選出 50 篇相關文章 (Sparse Retrieval)。
    2. 再用 Embedding 模型計算這 50 篇與 Query 的語意相似度 (Dense Retrieval)。
    3. 重新排序，回傳最終的 top_k。
    """
    def __init__(self, chunks, language="en"):
        self.language = language
        self.bm25 = BM25Retriever(chunks, language)
        
        # 設定 Embedding 模型
        # 但在程式執行時，會自動檢查模型是否存在
        self.embed_model = os.environ.get("EMBED_MODEL", "qwen3-embedding:0.6b")
        self._check_model_availability()

    def _check_model_availability(self):
        """檢查 Ollama 裡是否有指定的模型，沒有的話自動切換，避免 Crash。"""
        try:
            resp = self.client.list()
            models = [m['name'].split(':')[0] for m in resp.get('models', [])]
            target = self.embed_model.split(':')[0]
            
            if target not in models:
                print(f"[Warning] Model {self.embed_model} not found.")
                # 嘗試尋找替代品
                for m in models:
                    if "embed" in m or "bert" in m:
                        self.embed_model = m
                        print(f"[Fallback] Switching to {self.embed_model}")
                        return
                # 最糟情況：隨便抓一個
                if models:
                    self.embed_model = models[0]
                    print(f"[Fallback] Using first available model: {self.embed_model}")
        except Exception as e:
            print(f"[Error] Ollama connection failed: {e}")

    def _get_embedding(self, text):
        """呼叫 Ollama 取得向量"""
        try:
            # 呼叫 API
            response = self.client.embed(model=self.embed_model, input=text)
            # 處理回傳格式 (可能是 object 或是 dict)
            if isinstance(response, dict):
                return response.get("embeddings", [[]])[0]
            else:
                return response.embeddings[0]
        except Exception as e:
            print(f"[Error] Embedding failed: {e}")
            return None

    def _cosine_similarity(self, vec1, vec2):
        """計算餘弦相似度"""
        if not vec1 or not vec2:
            return 0.0
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def retrieve(self, query, top_k=10):
        # 步驟 1: BM25 廣度搜尋 (取出 50 篇候選)
        # 為什麼是 50？因為我們希望 Recall (召回率) 夠高，不要漏掉潛在答案
        candidates = self.bm25.retrieve_candidates(query, top_k=50)
        
        # 步驟 2: 計算 Query 的向量
        query_vec = self._get_embedding(query)
        
        # 如果 Embedding 失敗 (例如 Ollama 掛了)，直接回傳 BM25 的結果 (保命機制)
        if query_vec is None:
            return candidates[:top_k]

        # 步驟 3: 對候選文章進行 Rerank (重排序)
        reranked_results = []
        for chunk in candidates:
            # 這裡我們即時計算 chunk 的向量
            # 因為只有 50 篇，速度會很快，不需要預先算好存起來
            chunk_vec = self._get_embedding(chunk["page_content"])
            
            if chunk_vec:
                score = self._cosine_similarity(query_vec, chunk_vec)
                reranked_results.append((score, chunk))
            else:
                # 如果算不出向量，給個 0 分放在最後
                reranked_results.append((0.0, chunk))

        # 步驟 4: 依照相似度分數由高到低排序
        reranked_results.sort(key=lambda x: x[0], reverse=True)
        
        # 只回傳 chunk 本體 (去掉分數)
        final_chunks = [item[1] for item in reranked_results[:top_k]]
        
        return final_chunks



def create_retriever(chunks, language):
    """Creates a BM25 retriever from document chunks."""
    return BM25Retriever(chunks, language)