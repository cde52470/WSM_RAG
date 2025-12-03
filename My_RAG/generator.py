import os
from ollama import Client


def generate_answer(query, context_chunks, language="en", ollama_client=None):
    """
    Generates an answer using a concise prompt that asks the LLM to cite sources.
    """
    if ollama_client is None:
        raise ValueError("Ollama client must be provided.")

    # Number the contexts for citation
    numbered_contexts = []
    for idx, chunk in enumerate(context_chunks, start=1):
        numbered_contexts.append(f"[{idx}] {chunk['page_content']}")
    context = "\n\n".join(numbered_contexts)

    prompt = (
        "You are an assistant for question-answering tasks.\n"
        "You will be given a question and several numbered context snippets.\n"
        "Your job is to answer the question **only** using the information in the context.\n"
        "- If the context does not contain enough information, explicitly say that you don't know.\n"
        "- Keep the main answer within three sentences and be concise.\n"
        "- After your answer, add one line starting with 'Sources:' and list the ids of the snippets you used, "
        "for example: 'Sources: [1], [3]'.\n"
        "Do not fabricate information that is not supported by the context.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}\n\n"
        "Answer:\n"
    )

    try:
        model_name = os.getenv("GENERATOR_MODEL", "granite4:3b")
        response = ollama_client.generate(
            model=model_name,
            prompt=prompt,
            stream=False,
        )
        return response.get("response", "No response from model.")
    except Exception as e:
        return f"Error using Ollama Python client: {e}"


if __name__ == "__main__":
    # Simple local test
    # Ensure OLLAMA_HOST is set or Ollama is running on the default host
    class MockClient:
        def generate(self, model, prompt, stream):
            print("--- Mock Generate Call ---")
            print(f"Model: {model}")
            print(f"Prompt:\n{prompt}")
            return {"response": "This is a mock answer. Sources: [1], [2]"}

    query = "What is the capital of France?"
    context_chunks = [
        {"page_content": "France is a country in Europe. Its capital is Paris."},
        {"page_content": "The Eiffel Tower is located in Paris, the capital city of France."}
    ]
    print(generate_answer(query, context_chunks, ollama_client=MockClient()))