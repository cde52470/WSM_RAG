import os
from tqdm import tqdm
from utils import load_jsonl, save_jsonl
from chunker import chunk_documents
from retriever import create_retriever
from generator import generate_answer
from ollama import Client
import argparse

# --- [NEW] Multi-Query Generation Function ---
def generate_multiple_queries(original_query: str, ollama_client: Client) -> list[str]:
    """Generates variations of the original query using an LLM."""
    prompt = f"""You are a helpful assistant. Your task is to generate 3 different versions of the given user question to retrieve relevant documents from a vector database.
By generating multiple perspectives on the user question, we can help the user overcome some of the limitations of the distance-based similarity search.
Provide these alternative questions separated by newlines. Only provide the questions, no other text.

Original question: {original_query}"""

    try:
        # Use a smaller, faster model for this task if available
        model_name = os.getenv("REWRITER_MODEL", "gemma:2b")
        response = ollama_client.generate(model=model_name, prompt=prompt, stream=False)
        generated_text = response.get("response", "")
        # Split the response by newlines and filter out any empty strings
        queries = [q.strip() for q in generated_text.split('\n') if q.strip()]
        # Add the original query to the list
        queries.insert(0, original_query)
        return list(set(queries)) # Return unique queries
    except Exception as e:
        print(f"Warning: Failed to generate multiple queries, using original query only. Error: {e}")
        return [original_query]
# --- [END NEW] ---

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

    retriever = create_retriever(chunks, language)
    print("Retriever created successfully.")


    for query in tqdm(queries, desc="Processing Queries"):
        original_query_text = query['query']['content']
        
        # --- [MODIFIED] Multi-Query Retrieval ---
        # 4a. Generate multiple queries
        all_queries = generate_multiple_queries(original_query_text, ollama_client)
        # print(f"\nGenerated {len(all_queries)} queries: {all_queries}")

        # 4b. Retrieve for all queries and de-duplicate chunks
        all_retrieved_chunks = []
        for q_text in all_queries:
            retrieved = retriever.retrieve(q_text, top_k=3)
            all_retrieved_chunks.extend(retrieved)
        
        # De-duplicate chunks based on page_content
        unique_chunks_dict = {chunk['page_content']: chunk for chunk in all_retrieved_chunks}
        final_chunks = list(unique_chunks_dict.values())
        # print(f"Retrieved {len(final_chunks)} unique chunks.")
        # --- [END MODIFIED] ---

        # 5. Generate Answer
        # Use original query for answer generation
        answer = generate_answer(original_query_text, final_chunks, ollama_client)

        query["prediction"]["content"] = answer
        # Store all unique retrieved page contents as references
        query["prediction"]["references"] = [chunk['page_content'] for chunk in final_chunks]

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
