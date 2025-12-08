from ollama import Client
from pathlib import Path
import yaml

_OLLAMA_CLIENT = None
_OLLAMA_CLIENT_INITIALIZED = False

def load_ollama_config() -> dict:
    configs_folder = Path(__file__).parent.parent / "configs"
    config_paths = [
        configs_folder / "config_local.yaml",
        configs_folder / "config_submit.yaml",
    ]
    config = {}
    for path in config_paths:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}
            config = raw.get("ollama", {})
            break

    # 如果兩個檔都沒有，給一個本機預設值，方便你在自己電腦測試
    if not config:
        config = {
            "host": "http://127.0.0.1:11434",
            "model": "granite4:3b",
        }

    # 至少要有 model，host 可以之後再補
    if "model" not in config:
        config["model"] = "granite4:3b"

    return config

def get_ollama_client() -> Client:
    """
    按照下面順序嘗試連線：
    1. config 裡指定的 host（支援單一字串或 list）
    2. 預設候選 host：
       - http://ollama-gateway:11434
       - http://ollama:11434
       - http://localhost:11434
       - http://127.0.0.1:11434

    ✅ 第一次呼叫時才會真的去「掃 host」
    ✅ 之後呼叫全部直接回傳同一個 client，不再重試
    """
    global _OLLAMA_CLIENT, _OLLAMA_CLIENT_INITIALIZED

    # 如果已經初始化過，就直接用快取的 client（不再打任何 request）
    if _OLLAMA_CLIENT_INITIALIZED:
        return _OLLAMA_CLIENT

    cfg = load_ollama_config()

    candidate_hosts = []

    # 1) config 中的 host（若 server 端有特別設定，會放在這裡）
    if "host" in cfg and cfg["host"]:
        if isinstance(cfg["host"], str):
            candidate_hosts.append(cfg["host"])
        elif isinstance(cfg["host"], (list, tuple)):
            candidate_hosts.extend(cfg["host"])

    # 2) 加上內建預設候選 host（避免重複）
    default_hosts = [
        "http://ollama-gateway:11434",
        "http://ollama:11434",
        "http://localhost:11434",
        "http://127.0.0.1:11434",
    ]
    for h in default_hosts:
        if h not in candidate_hosts:
            candidate_hosts.append(h)

    last_error = None
    for host in candidate_hosts:
        try:
            client = Client(host=host)
            # 用 list() 當健康檢查，成功就視為這個 host 可用
            client.list()
            print(f"[INFO] Connected to Ollama at {host}")

            # 設定快取，只做一次
            _OLLAMA_CLIENT = client
            _OLLAMA_CLIENT_INITIALIZED = True
            return client

        except Exception as e:
            print(
                f"[WARN] Failed to connect to Ollama at {host}. "
                f"Error: {e}"
            )
            last_error = e

    # 掃完全部 host 都失敗，標記為已初始化，避免之後一直重試
    _OLLAMA_CLIENT = None
    _OLLAMA_CLIENT_INITIALIZED = True
    raise ConnectionError(
        f"Failed to connect to any Ollama host. Last error: {last_error}"
    )


def is_contains_chinese(strs):
    """檢查字串是否包含中文字元"""
    for _char in strs:
        if '\u4e00' <= _char <= '\u9fff':
            return True
    return False

def _parse_model_output(response_text: str, language: str) -> str:
    """
    解析模型輸出，移除思考過程，只保留最終答案。
    """
    # 定義要捕捉的標籤
    tags = ["Answer:", "最終答案：", "Final Answer:", "回答："]
    
    # 1. 嘗試尋找分割點
    content = response_text.strip()
    for tag in tags:
        if tag in content:
            # 取 tag 之後的所有文字
            content = content.split(tag)[-1].strip()
            return content
            
    # 2. 如果沒找到 tag (模型沒乖乖聽話)，嘗試用換行符號猜測
    # 通常思考過程長，答案短，或是思考在第一段。
    # 這裡採取保守策略：如果沒 tag，就回傳全部，避免切錯。
    return content

def generate_answer(query, context_chunks):
    # 1. 準備 Context
    context = "\n\n".join([chunk['page_content'] for chunk in context_chunks])
    
    # 2. 準備 Prompt (加入 CoT 與格式要求)
    if is_contains_chinese(query):
        # 【中文 Prompt：強調推論與格式】
        prompt = (
            "你是一个严谨的问答助手。请仅根据提供的「参考内容」回答问题。\n"
            "若参考内容中没有答案，请直接说「我不知道」，不可编造。\n\n"
            "严格比对公司名称与年份，不要混淆不同公司／年份的数据\n"
            "如果 context 提到股息或影响，务必写出，不要遗漏\n"
            "请严格遵守以下输出格式：\n"
            "思考过程：<请在此简短分析参考内容与问题的关联>\n"
            "最终答案：<请在此给出最终的简体中文回答，不超过三句话>\n\n"
            f"参考内容 (Context):\n{context}\n\n"
            f"使用者问题 (Question): {query}\n"
        )
    else:
        # 【英文 Prompt：強調 Reasoning 與 Format】
        prompt = (
            "You are a strict assistant. Answer the question based ONLY on the provided context.\n"
            "If the answer is not in the context, say 'I don't know'. Do not hallucinate.\n\n"
            "Verify company names and years exactly match the question; do not mix different entities.\n"
            "If context mentions dividends or their impacts, include them and don't omit available facts.\n"
            "Please follow this format strictly:\n"
            "Thinking: <Briefly analyze the context and reasoning here>\n"
            "Answer: <Provide the final concise answer here, max 3 sentences>\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
        )


    cfg = load_ollama_config()
    model = cfg.get("model", "granite4:3b")
    
    try:
        client = get_ollama_client()
        response = client.generate(model=model, prompt=prompt)
        raw_output = response["response"]

        lang = "zh" if is_contains_chinese(query) else "en"
        final_answer = _parse_model_output(raw_output, lang)
        return final_answer

    except Exception as e:
        return f"Error generating answer: {e}"
    # 3. 呼叫模型
    #ollama_config = load_ollama_config()
    # 優先使用 config 設定，若無則 fallback 到預設值
    #host = ollama_config.get("host", "http://localhost:11434")
    # 建議之後換成 qwen2.5:3b 以獲得更好的中文效果
    #model = "granite4:3b" # 或者保留原本的 "granite4:3b"
    
    try:
        client = Client(host=host)
        response = client.generate(model=model, prompt=prompt)
        raw_output = response["response"]
        
        # 4. 解析輸出 (只回傳 Answer 部分)
        final_answer = _parse_model_output(raw_output, "zh" if is_contains_chinese(query) else "en")
        return final_answer
        
    except Exception as e:
        print(f"Generate Error: {e}")
        return "Sorry, generation failed."    

if __name__ == "__main__":
    # 測試程式
    query_zh = "法國的首都在哪裡？"
    chunks = [
        {"page_content": "法國（France），全名法蘭西共和國。"},
        {"page_content": "巴黎（Paris）是法國的首都及最大都市。"}
    ]
    
    print("Testing Chinese Query...")
    ans = generate_answer(query_zh, chunks)
    print(f"Parsed Answer: {ans}")
    
    print("\nTesting English Query...")
    ans_en = generate_answer("What is the capital of France?", chunks)
    print(f"Parsed Answer: {ans_en}")

# def generate_answer(query, context_chunks, language="en"):
#    context = "\n\n".join([chunk["page_content"] for chunk in context_chunks])
    # ENGLISH_prompt
#    if language == "en":
#        prompt = f"""You are a careful assistant. Follow these rules:
#- Only use information from context; if not present, answer "I don't know."
#- If context mentions dividends or their impacts, include them and don't omit available facts.

#Question: {query}
#Context:
#{context}

#Answer:"""

    # CHINESE_prompt
#    else:
#        prompt = f"""你是一個謹慎的助理，請只使用下方 context 回答，遵守：
#- 只用 context 內容，沒有就回答「我不知道」。
#- 嚴格比對公司名稱與年份，不要混淆不同公司／年份的數據。
#- 如果 context 提到股息或影響，務必寫出，不要遺漏。

#Question: {query}
#Context:
#{context}

#Answer:
#"""

#    ollama_config = load_ollama_config()
#    client = Client(host=ollama_config["host"])
#    response = client.generate(model=ollama_config["model"], prompt=prompt)
#    return response["response"]


#if __name__ == "__main__":
    # test the function
#    query = "What is the capital of France?"
#    context_chunks = [
#        {"page_content": "France is a country in Europe. Its capital is Paris."},
#        {
#            "page_content": "The Eiffel Tower is located in Paris, the capital city of France."
#        },
#    ]
#    answer = generate_answer(query, context_chunks)
#    print("Generated Answer:", answer)

