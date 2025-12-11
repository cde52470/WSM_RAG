from sentence_transformers import SentenceTransformer
import re


class chunker:
    def __init__(self, model_name="BAAI/bge-m3", threshold=0.5):
        self.model = SentenceTransformer(model_name)
        self.threshold = threshold  # 低於 threshold 的句子會被合併

    def spilt_txt_into_sentences(self, text, chunk_size=500, chunk_overlap=50):
        sentences = re.split(
            r"(?<=[。！？.!?\n])", text
        )  # /?<= 代表切完後要把標點符號留在前一個句子
        return [s for s in sentences if s.strip()]

    def chunk_documents(self, docs, language):
        all_chunks = []
        for doc in docs:
            content = doc.get("content", "")
            if not content.strip():
                continue

            sentences = [
                s for s in self.spilt_txt_into_sentences(content) if s.strip()
            ]  # 先把抓下來的文章切成句子
            if not sentences:
                continue

            embeddings = self.model.encode(
                sentences, convert_to_tensor=True
            )  # 把句子轉成向量
