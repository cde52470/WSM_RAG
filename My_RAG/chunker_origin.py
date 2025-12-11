import re


def chunk_documents(docs, language, chunk_size=800, chunk_overlap=300):
    chunks = []
    language = str(language).lower().strip()
    chunk_overlap = max(
        0, min(chunk_overlap, chunk_size)
    )  # 確保 overlap：不能小於 0，且不能大於區塊總大小 (size)

    # 根據語言選擇適當的分句模式
    if language == "en":
        pattern = r"(\n\n|(?<=[.!?])\s+)"
    else:
        pattern = r"(\n\n|(?<=[。！？；]))"

    for doc in docs:
        text = doc.get("content")
        if (
            not isinstance(text, str) or not text.strip()
        ):  # if text is None or not a string or empty
            continue

        doc_lang = doc.get("language", "en").lower().strip()
        if doc_lang != language:
            continue

        # Split text by pattern, keep delimiters, and clean up.
        parts = re.split(pattern, text)
        sentences = []  # store sentences after splitting
        for i in range(
            0, len(parts), 2
        ):  # parts[i] is content, parts[i+1] is delimiter that's why step by 2
            sentence = parts[i] + (parts[i + 1] if i + 1 < len(parts) else "")
            if sentence.strip():
                sentences.append(sentence)

        if not sentences:
            continue

        chunk_index = 0
        current_chunk_sentences = []
        current_len = 0

        for sentence in sentences:
            sent_len = len(sentence)

            if current_chunk_sentences and current_len + sent_len > chunk_size:
                chunk_text = "".join(
                    current_chunk_sentences
                ).strip()  # combine 2 of them for a new one
                if chunk_text:
                    chunk_metadata = doc.copy()
                    chunk_metadata.pop("content", None)
                    chunk_metadata["chunk_index"] = chunk_index
                    chunks.append(
                        {"page_content": chunk_text, "metadata": chunk_metadata}
                    )
                    chunk_index += 1

                # Sentence-aware overlap
                overlap_sents = []
                overlap_len = 0
                for head_of_next_round in reversed(current_chunk_sentences):
                    if overlap_len < chunk_overlap:
                        overlap_sents.insert(0, head_of_next_round)
                        overlap_len += len(head_of_next_round)
                    else:
                        break

                # The new chunk starts with the overlap
                current_chunk_sentences = overlap_sents
                current_len = overlap_len

            # Add current sentence to the chunk-in-progress
            current_chunk_sentences.append(sentence)
            current_len += len(sentence)

        # Add the final chunk if any sentences are left
        if current_chunk_sentences:
            chunk_text = "".join(current_chunk_sentences).strip()
            if chunk_text:
                chunk_metadata = doc.copy()
                chunk_metadata.pop("content", None)
                chunk_metadata["chunk_index"] = chunk_index
                chunks.append({"page_content": chunk_text, "metadata": chunk_metadata})

    return chunks
