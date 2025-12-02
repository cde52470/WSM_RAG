from tqdm import tqdm
from utils import load_jsonl, save_jsonl
from chunker import chunk_documents
from retriever import create_retriever
from generator import generate_answer
from ollama import Client
import argparse

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
    # 3. Create Retriever
    print("Creating retriever...")

    # --- Ollama Client Instantiation ---
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


    for query in tqdm(queries, desc="Processing Queries"):
        # 4. Retrieve relevant chunks
        query_text = query['query']['content']
        # print(f"\nRetrieving chunks for query: '{query_text}'")
        retrieved_chunks = retriever.retrieve(query_text, top_k=3)
        # print(f"Retrieved {len(retrieved_chunks)} chunks.")

        # 5. Generate Answer
        # print("Generating answer...")
        answer = generate_answer(query_text, retrieved_chunks, ollama_client)

        query["prediction"]["content"] = answer
        query["prediction"]["references"] = [retrieved_chunks[0]['page_content']]

    save_jsonl(output_path, queries)
    print("Predictions saved at '{}'".format(output_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', help='Path to the query file')
    parser.add_argument('--docs_path', help='Path to the documents file')
    parser.add_argument('--language', help='Language to filter queries (zh or en), if not specified, process all')
    parser.add_argument('--output', help='Path to the output file')
    args = parser.parse_args()
    main(args.query_path, args.docs_path, args.language, args.output)
