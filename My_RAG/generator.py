# generator.py
import os
from ollama import Client


def _get_ollama_client():
    """
    建立一個 Ollama Client。
    - 在正式評分環境中，建議 host 使用 http://ollama-gateway:11434
    - 若本地測試有自己的 OLLAMA_HOST，也可以用環境變數覆寫。
    """
    host = os.environ.get("OLLAMA_HOST", "http://ollama-gateway:11434")
    return Client(host=host)


def generate_answer(query, context_chunks):
    """
    使用 granite4:3b 和較嚴謹的 prompt，依據 context_chunks 回答 query。

    Args:
        query (str): 使用者問題。
        context_chunks (list[dict]): 由 retriever 傳回的 chunks，每個有 'page_content'。

    Returns:
        str: LLM 生成的答案。
    """
    # 把每個 chunk 標上編號，方便在答案中引用
    # 例如：
    # [1] ...chunk1 內容...
    # [2] ...chunk2 內容...
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
        client = _get_ollama_client()
        response = client.generate(
            model="granite4:3b",
            prompt=prompt,
            stream=False,
        )
        # Ollama Python client 回傳 dict，文字在 'response'
        return response.get("response", "No response from model.")
    except Exception as e:
        return f"Error using Ollama Python client: {e}"


if __name__ == "__main__":
    # 簡單本地測試
    query = "What is the capital of France?"
    context_chunks = [
        {"page_content": "France is a country in Europe. Its capital is Paris."},
        {"page_content": "The Eiffel Tower is located in Paris, the capital city of France."}
    ]
    print(generate_answer(query, context_chunks))
