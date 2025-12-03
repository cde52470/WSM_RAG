from tqdm import tqdm
from utils import load_jsonl, save_jsonl
from chunker import chunk_documents
from retriever import create_retriever
from generator import generate_answer
import argparse
import re
import jieba
from ollama import Client


def _split_sentences(text: str, language: str):
    """Very simple sentence splitter for zh / en.

    - zh：用 。！？ 當句尾，保留標點。
    - en：用 .!? + 空白 切句。
    """
    if not text:
        return []

    if language == "zh":
        parts = re.split(r"([。！？])", text)
        sentences = []
        for i in range(0, len(parts) - 1, 2):
            sent = (parts[i] + parts[i + 1]).strip()
            if sent:
                sentences.append(sent)
        # 最後可能有沒標點的殘句
        if len(parts) % 2 == 1 and parts[-1].strip():
            sentences.append(parts[-1].strip())
        return sentences
    else:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return [s.strip() for s in sentences if s.strip()]


def _select_reference_sentences(query_text: str, retrieved_chunks, language: str, max_refs: int = 5):
    """從 retrieved chunks 中挑出跟 query 最相關的幾句 sentence 當作 references。

    做法：
    1. 把每個 chunk 切成句子。
    2. 用「query 與 sentence 的 token 重疊比例」當 score。
    3. 取分數最高的前 max_refs 句，去掉重複。
    """
    if not retrieved_chunks:
        return []

    # 1) 收集所有候選句子
    candidate_sentences = []
    for chunk in retrieved_chunks:
        text = chunk.get("page_content", "")
        for sent in _split_sentences(text, language):
            if sent.strip():
                candidate_sentences.append(sent.strip())

    if not candidate_sentences:
        return []

    # 2) 準備 query token set
    if language == "zh":
        q_tokens = set(token for token in jieba.cut(query_text) if token.strip())

        def sent_score(sent: str):
            s_tokens = set(token for token in jieba.cut(sent) if token.strip())
            if not s_tokens:
                return 0.0
            return len(q_tokens & s_tokens) / (len(s_tokens) + 1e-8)
    else:
        word_re = re.compile(r"\w+")
        q_tokens = set(w.lower() for w in word_re.findall(query_text))

        def sent_score(sent: str):
            s_tokens = set(w.lower() for w in word_re.findall(sent))
            if not s_tokens:
                return 0.0
            return len(q_tokens & s_tokens) / (len(s_tokens) + 1e-8)

    # 3) 計算分數並排序
    scored = [(sent, sent_score(sent)) for sent in candidate_sentences]
    scored.sort(key=lambda x: x[1], reverse=True)

    # 4) 取前 max_refs 個不重複 sentence
    references = []
    for sent, score in scored:
        if sent not in references:
            references.append(sent)
        if len(references) >= max_refs:
            break

    return references


def main(query_path, docs_path, language, output_path):
    # 1. Load Data
    print("Loading documents...")
    docs_for_chunking = load_jsonl(docs_path)
    queries = load_jsonl(query_path)
    print(f"Loaded {len(docs_for_chunking)} documents.")
    print(f"Loaded {len(queries)} queries.")

    # 2. Chunk Documents
    print("Chunking documents...")
    chunks = chunk_documents(docs_for_chunking, language)
    print(f"Created {len(chunks)} chunks.")

    # 3. Create Retriever
    print("Creating retriever...")
    retriever = create_retriever(chunks, language)
    print("Retriever created successfully.")

    hosts_to_try = [
        "http://ollama-gateway:11434",  # Submission host
        "http://ollama:11434",          # Local Docker host
        "http://localhost:11434"        # Local Conda host
    ]
    ollama_client = None
    for host in hosts_to_try:
        try:
            temp_client = Client(host=host)
            temp_client.list() # Test connectivity
            ollama_client = temp_client
            print(f"Connected to Ollama at {host}")
            break # Successfully connected, exit loop
        except Exception as e:
            print(f"Warning: Failed to connect to Ollama at {host}. Trying next host. Error: {e}")
            continue
    
    if ollama_client is None:
        raise ConnectionError("Failed to connect to any Ollama host.")
    # --- End Ollama Client Instantiation ---

    retriever = create_retriever(chunks, language, client=ollama_client)
    print("Retriever created successfully.")

    # 4. For each query: retrieve -> generate -> build prediction
    for query in tqdm(queries, desc="Processing Queries"):
        query_text = query["query"]["content"]

        # 4.1 hybrid BM25 + embedding retrieve
        retrieved_chunks = retriever.retrieve(query_text)

        # 4.2 用 retrieved chunks 生成答案
        answer = generate_answer(query_text, retrieved_chunks)

        # 確保 prediction 欄位存在
        if "prediction" not in query or not isinstance(query["prediction"], dict):
            query["prediction"] = {}

        query["prediction"]["content"] = answer

        # 4.3 選最相關的句子當 reference（比整段 chunk 精細很多）
        reference_sentences = _select_reference_sentences(query_text, retrieved_chunks, language, max_refs=5)
        query["prediction"]["references"] = reference_sentences

    save_jsonl(output_path, queries)
    print(f"Predictions saved at '{output_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', help='Path to the query file')
    parser.add_argument('--docs_path', help='Path to the documents file')
    parser.add_argument('--language', help='Language to filter queries (zh or en), if not specified, process all')
    parser.add_argument('--output', help='Path to the output file')
    args = parser.parse_args()
    main(args.query_path, args.docs_path, args.language, args.output)
