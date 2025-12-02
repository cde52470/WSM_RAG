from rank_bm25 import BM25Okapi
import jieba
import numpy as np
import re


class BM25Retriever:
    def __init__(self, chunks, language="en"):
        self.chunks = chunks
        self.language = language
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

    def retrieve(self, query, top_k=10):
        if self.language == "zh":
            tokenized_query = list(self._tokenizer.cut(query))
        else:
            tokenized_query = query.split(" ")

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

        top_k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [self.chunks[i] for i in top_indices]


def create_retriever(chunks, language):
    """Creates a BM25 retriever from document chunks."""
    return BM25Retriever(chunks, language)
