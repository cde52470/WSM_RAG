import os
from tqdm import tqdm
from .utils import load_jsonl, save_jsonl
from .chunker import chunk_documents
from .retriever import create_retriever
from .generator import generate_answer
import argparse


class RAGPipeline:
    def __init__(self, docs_path, language):
        print("Loading documents...")
        self.docs_for_chunking = load_jsonl(docs_path)
        print(f"Loaded {len(self.docs_for_chunking)} documents.")

        print("Chunking documents...")
        self.chunks = chunk_documents(self.docs_for_chunking, language)
        if len(self.chunks) == 0:
            raise ValueError(
                f"Error: No chunks created! Please check if your document language matches input language '{language}'."
            )
        print(f"Created {len(self.chunks)} chunks.")

        print("Creating retriever...")
        self.retriever = create_retriever(self.chunks, language)
        print("Retriever created successfully.")
        self.language = language

    def run(self, query_path, output_path):
        queries = load_jsonl(query_path)
        print(f"Loaded {len(queries)} queries.")

        for query in tqdm(queries, desc="Processing Queries"):
            query_text = query["query"]["content"]
            qLanguage = query.get("language", self.language) or "en"
            
            retrieved_chunks = self.retriever.retrieve(query_text, top_k=10)
            
            answer = generate_answer(query_text, retrieved_chunks, qLanguage)
            prediction = query.setdefault("prediction", {})
            prediction["content"] = answer
            prediction["references"] = [chunk["page_content"] for chunk in retrieved_chunks]
            
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_jsonl(output_path, queries)
        print("Predictions saved at '{}'".format(output_path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_path", help="Path to the query file")
    parser.add_argument("--docs_path", help="Path to the documents file")
    parser.add_argument(
        "--language",
        help="Language to filter queries (zh or en), if not specified, process all",
    )
    parser.add_argument("--output", help="Path to the output file")
    args = parser.parse_args()

    try:
        pipeline = RAGPipeline(args.docs_path, args.language)
        pipeline.run(args.query_path, args.output)
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    main()
