from ollama import Client
import os

def generate_answer(query, context_chunks):
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
    hosts_to_try = [
        "http://ollama-gateway:11434",  # Submission host
        "http://ollama:11434",          # Local Docker host
        "http://localhost:11434"        # Local Conda host
    ]
    model_name = os.getenv("GENERATOR_MODEL", "granite4:3b")
    last_error = None

    for host in hosts_to_try:
        try:
            client = Client(host=host)
            # Use a lightweight call to check for connectivity before generating
            client.list()
            # If connectivity is confirmed, proceed with generation
            response = client.generate(model=model_name, prompt=prompt, stream=False)
            return response.get("response", "No response from model.")
        except Exception as e:
            # print(f"Info: Failed to connect to {host}. Trying next host. Error: {e}")
            last_error = e
            continue  # Try the next host in the list

    # If all hosts fail, return the last error
    return f"Error: Could not connect to any Ollama host. Last error: {last_error}"


if __name__ == "__main__":
    # test the function
    query = "What is the capital of France?"
    context_chunks = [
        {"page_content": "France is a country in Europe. Its capital is Paris."},
        {"page_content": "The Eiffel Tower is located in Paris, the capital city of France."}
    ]
    answer = generate_answer(query, context_chunks)
    print("Generated Answer:", answer)