# from ollama import Client # Removed as client is passed
import os # Keep os for getenv

def generate_answer(query, context_chunks, ollama_client): # Added ollama_client argument
    context = "\n\n".join([chunk['page_content'] for chunk in context_chunks])
    prompt = f"""You are a helpful assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
If the answer is not in the context, just say that you don't know.
Keep the answer concise and strictly based on the provided context.

<context>
{context}
</context>

<question>
{query}
</question>

Answer:"""
    try:
        model_name = os.getenv("GENERATOR_MODEL", "granite4:3b")
        response = ollama_client.generate(model=model_name, prompt=prompt, stream=False) # Use passed client
        return response.get("response", "No response from model.")
    except Exception as e:
        return f"Error using Ollama Python client: {e}"


if __name__ == "__main__":
    # test the function
    query = "What is the capital of France?"
    context_chunks = [
        {"page_content": "France is a country in Europe. Its capital is Paris."},
        {"page_content": "The Eiffel Tower is located in Paris, the capital city of France."}
    ]
    answer = generate_answer(query, context_chunks)
    print("Generated Answer:", answer)