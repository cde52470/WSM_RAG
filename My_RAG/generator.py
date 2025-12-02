from ollama import Client
from pathlib import Path
import yaml


def load_ollama_config() -> dict:
    configs_folder = Path(__file__).parent.parent / "configs"
    config_paths = [
        configs_folder / "config_local.yaml",
        configs_folder / "config_submit.yaml",
    ]
    config_path = None
    for path in config_paths:
        if path.exists():
            config_path = path
            break

    if config_path is None:
        raise FileNotFoundError("No configuration file found.")

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    assert "ollama" in config, "Ollama configuration not found in config file."
    assert "host" in config["ollama"], "Ollama host not specified in config file."
    assert "model" in config["ollama"], "Ollama model not specified in config file."
    return config["ollama"]


def generate_answer(query, context_chunks, language="en"):
    context = "\n\n".join([chunk["page_content"] for chunk in context_chunks])
    # ENGLISH_prompt
    if language == "en":
        prompt = f"""You are a careful assistant. Follow these rules:
- Only use information from context; if not present, answer "I don't know."
- Verify company names and years exactly match the question; do not mix different entities.
- If context mentions dividends or their impacts, include them and don't omit available facts.

Question: {query}
Context:
{context}

Answer:"""

    # CHINESE_prompt
    else:
        prompt = f"""你是一個謹慎的助理，請只使用下方 context 回答，遵守：
- 只用 context 內容，沒有就回答「我不知道」。
- 嚴格比對公司名稱與年份，不要混淆不同公司／年份的數據。
- 如果 context 提到股息或影響，務必寫出，不要遺漏。

Question: {query}
Context:
{context}

Answer:
"""

    ollama_config = load_ollama_config()
    client = Client(host=ollama_config["host"])
    response = client.generate(model=ollama_config["model"], prompt=prompt)
    return response["response"]


if __name__ == "__main__":
    # test the function
    query = "What is the capital of France?"
    context_chunks = [
        {"page_content": "France is a country in Europe. Its capital is Paris."},
        {
            "page_content": "The Eiffel Tower is located in Paris, the capital city of France."
        },
    ]
    answer = generate_answer(query, context_chunks)
    print("Generated Answer:", answer)
