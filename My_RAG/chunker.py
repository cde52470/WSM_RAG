# chunker.py

def chunk_documents(docs, language, chunk_size=None, chunk_overlap=None):
    """
    將指定語言的文件依「字元數 + overlap」切成 chunks。

    這裡有做一點 language-aware：
    - 若未指定 chunk_size / chunk_overlap：
      - zh: chunk_size=800,  chunk_overlap=200
      - en: chunk_size=1200, chunk_overlap=200

    Args:
        docs (list[dict]): 來自 dragonball_docs.jsonl 的文件列表，
                           每筆至少有 'content' 和 'language'。
        language (str): 'zh' 或 'en'。
        chunk_size (int | None): 每個 chunk 的長度（字元）。None 則用預設。
        chunk_overlap (int | None): chunk 之間重疊的字元數。None 則用預設。

    Returns:
        list[dict]: 每個元素包含：
            - 'page_content': chunk 的文字
            - 'metadata': 原始文件的 meta + 'chunk_index'
    """
    # 1) 設定 language-aware 的預設值
    if chunk_size is None or chunk_overlap is None:
        if language == "zh":
            default_size, default_overlap = 800, 200
        else:  # 預設當英文處理
            default_size, default_overlap = 1200, 200
        if chunk_size is None:
            chunk_size = default_size
        if chunk_overlap is None:
            chunk_overlap = default_overlap

    chunks = []
    for doc_index, doc in enumerate(docs):
        # 基本欄位檢查
        if 'content' not in doc or 'language' not in doc:
            continue
        if not isinstance(doc['content'], str):
            continue
        if doc['language'] != language:
            continue

        text = doc['content']
        text_len = len(text)

        start_index = 0
        chunk_count = 0

        # 避免 chunk_size == chunk_overlap 造成死循環
        step = max(1, chunk_size - chunk_overlap)

        while start_index < text_len:
            end_index = min(start_index + chunk_size, text_len)

            # 建立 metadata：保留原有欄位，但移除 content，並加上 chunk_index
            chunk_metadata = doc.copy()
            chunk_metadata.pop('content', None)
            chunk_metadata['chunk_index'] = chunk_count

            chunk = {
                'page_content': text[start_index:end_index],
                'metadata': chunk_metadata
            }
            chunks.append(chunk)

            start_index += step
            chunk_count += 1

    return chunks
