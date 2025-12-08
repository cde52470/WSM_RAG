from ollama import Client
from pathlib import Path
import yaml
import re

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
        return {"host": "http://ollama-gateway:11434", "model": "granite4:3b"}

    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    return config.get("ollama", {})

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

def generate_answer(query, context_chunks, ollama_client):
    # 1. 準備 Context
    context = "\n\n".join([chunk['page_content'] for chunk in context_chunks])
    
    # 2. 準備 Prompt (加入 CoT 與格式要求)
    if is_contains_chinese(query):
        # 【中文 Prompt：強調推論與格式】
        prompt = (
            "你是一个严谨的问答助手。请仅根据提供的「参考内容」回答问题。\n"
            "若参考内容中没有答案，请直接说「我不知道」，不可编造。\n\n"
            "请严格遵守以下指令：\n"
            "1. 严格比对公司名称与年份，不要混淆不同公司／年份的数据。\n"
            "2. 如果参考内容提到股息或影响，务必写出，不要遗漏。\n\n"
            "请严格遵守以下输出格式：\n"
            "思考过程：<请在此简短分析参考内容与问题的关联>\n"
            "最终答案：<请在此给出最终的简体中文回答，不超过三句话>\n\n"
            "### 范例 (Example) ###\n"
            "参考内容: 2023年，台积电 (TSMC) 的营收达到了700亿美元。然而在2022年，其营收仅为600亿。\n"
            "使用者问题: 台积电2023年的营收是多少？\n"
            "思考过程: 参考内容明确指出了台积电在2023年的营收数值。上下文中的年份和公司名称与问题匹配。\n"
            "最终答案: 台积电在2023年的营收达到了700亿美元。\n"
            "### 结束范例 ###\n\n"
            f"参考内容 (Context):\n{context}\n\n"
            f"使用者问题 (Question): {query}\n"
        )
    else:
        # 【英文 Prompt：強調 Reasoning 與 Format】
        prompt = (
            "You are a strict assistant. Answer the question based ONLY on the provided context.\n"
            "If the answer is not in the context, say 'I don't know'. Do not hallucinate.\n\n"
            "Please follow these instructions strictly:\n"
            "1. Verify company names and years exactly match the question; do not mix different entities.\n"
            "2. If context mentions dividends or their impacts, include them and don't omit available facts.\n\n"
            "Please follow this format strictly:\n"
            "Thinking: <Briefly analyze the context and reasoning here>\n"
            "Answer: <Provide the final concise answer here, max 3 sentences>\n\n"
            "### Example ###\n"
            "Context: In 2023, TSMC reported a revenue of $70 billion. This was an increase from $60 billion in 2022.\n"
            "Question: What was TSMC's revenue in 2023?\n"
            "Thinking: The context explicitly states the revenue for the year 2023 matching the company TSMC.\n"
            "Answer: TSMC's revenue in 2023 was $70 billion.\n"
            "### End Example ###\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n"
        )

    # 3. 呼叫模型
    # 優先使用 config 設定，若無則 fallback 到預設值
    ollama_config = load_ollama_config()
    model = "granite4:3b" # 或者保留原本的 "granite4:3b"
    
    try:
        # 直接使用傳入的 ollama_client
        response = ollama_client.generate(model=model, prompt=prompt)
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