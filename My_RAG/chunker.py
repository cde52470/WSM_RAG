import re
from typing import Any, Dict, List, Optional

def _validate_chunk_params(chunk_size: int, chunk_overlap: int) -> int:
    """Validate chunk configuration."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")
    return chunk_size - chunk_overlap

def _split_text_into_sentences(text: str) -> List[Dict[str, Any]]:
    """
    使用正則表達式將文本切分為句子，並保留每個句子的起始位置。
    支援中文 (。！？) 與英文 (.!?) 及換行符。
    """
    # 說明：
    # [^。！？.!?\n]+  -> 匹配非標點符號的連續字元
    # [。！？.!?\n]* -> 匹配跟隨在後的標點符號 (包含換行)
    # 這樣可以確保標點符號會黏在句子後面，不會被切掉
    sentence_pattern = re.compile(r'([^。！？.!?\n]+[。！？.!?\n]*)')
    
    sentences = []
    # 使用 finditer 可以直接取得匹配文字的 start 和 end 索引
    for match in sentence_pattern.finditer(text):
        sentences.append({
            "text": match.group(),
            "start": match.start(),
            "end": match.end()
        })
    
    # 如果文本沒有標點符號 (例如只有一句話)，直接回傳整個文本
    if not sentences and text:
        sentences.append({"text": text, "start": 0, "end": len(text)})
        
    return sentences

def chunk_documents(
    docs: List[Dict[str, Any]],
    language: Optional[str] = None,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict[str, Any]]:
    """
    優化版的切分函數：基於句子邊界進行切分，避免切斷語意。
    """
    _validate_chunk_params(chunk_size, chunk_overlap)
    chunks: List[Dict[str, Any]] = []

    for doc_index, doc in enumerate(docs):
        text = doc.get("content")
        lang = doc.get("language")

        # 1. 基礎檢查與過濾
        if not isinstance(text, str) or not text.strip():
            continue
        if language and lang != language:
            continue

        text = text.strip()
        
        # 2. 先將整篇文章切成「句子」清單
        sentences = _split_text_into_sentences(text)
        
        current_chunk_sentences = []
        current_chunk_len = 0
        chunk_count = 0
        
        i = 0
        while i < len(sentences):
            sent = sentences[i]
            sent_len = len(sent["text"])
            
            # 3. 判斷是否加入當前 Chunk
            # 如果加入這句會爆掉 chunk_size，且當前 chunk 不為空，則先結算當前 chunk
            if current_chunk_len + sent_len > chunk_size and current_chunk_sentences:
                # --- 結算目前的 Chunk ---
                chunk_text = "".join([s["text"] for s in current_chunk_sentences])
                
                # 複製原始 Metadata
                chunk_metadata = {k: v for k, v in doc.items() if k != "content"}
                chunk_metadata.update({
                    "chunk_index": chunk_count,
                    "doc_index": doc_index,
                    "char_start": current_chunk_sentences[0]["start"],
                    "char_end": current_chunk_sentences[-1]["end"],
                })

                chunks.append({
                    "page_content": chunk_text,
                    "metadata": chunk_metadata
                })
                chunk_count += 1
                
                # --- 處理 Overlap (回溯) ---
                # 我們需要保留尾部的句子作為下一個 Chunk 的開頭 (Overlap)
                # 從後面往回算，直到湊滿 chunk_overlap 的長度
                overlap_len = 0
                new_start_idx = i  # 預設不重疊 (如果 overlap 設很小)
                
                for k in range(len(current_chunk_sentences) - 1, -1, -1):
                    overlap_len += len(current_chunk_sentences[k]["text"])
                    if overlap_len > chunk_overlap:
                        # 找到剛好超過 overlap 的點，這就是下一句的起點
                        new_start_idx = i - (len(current_chunk_sentences) - k) + 1
                        # 修正：確保至少會前進一步，避免無窮迴圈
                        new_start_idx = max(new_start_idx, i - len(current_chunk_sentences) + 1)
                        break
                
                # 如果 overlap 沒填滿整個 current chunk，我們就從計算出的位置重新開始
                # 如果 overlap 比整段還長 (罕見)，就只退一步
                if chunk_overlap > 0:
                    i = new_start_idx
                
                # 重置 Buffer
                current_chunk_sentences = []
                current_chunk_len = 0
                
                # 注意：這裡不 i+=1，因為我們要用新的 i 重新跑迴圈來判定當前句子
                continue

            # 4. 加入句子到 Buffer
            current_chunk_sentences.append(sent)
            current_chunk_len += sent_len
            i += 1
        
        # 5. 處理最後一個剩下的 Chunk
        if current_chunk_sentences:
            chunk_text = "".join([s["text"] for s in current_chunk_sentences])
            chunk_metadata = {k: v for k, v in doc.items() if k != "content"}
            chunk_metadata.update({
                "chunk_index": chunk_count,
                "doc_index": doc_index,
                "char_start": current_chunk_sentences[0]["start"],
                "char_end": current_chunk_sentences[-1]["end"],
            })
            chunks.append({
                "page_content": chunk_text,
                "metadata": chunk_metadata
            })

    return chunks
