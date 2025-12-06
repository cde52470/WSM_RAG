import re


def chunk_documents(docs, language):
    """Split documents into overlapping chunks on sentence-like boundaries."""
    chunk_overlap=300
    chunk_size=800
    chunks = []
    language = str(language).lower().strip()
    chunk_overlap = max(0, min(chunk_overlap, chunk_size))
    separators = (
        ["\n\n", "\n", ". ", "! ", "? ", "; "]
        if language == "en"
        else [
            "\n\n",
            "\n",
            "。",
            "！",
            "？",
            "；",
        ]
    )
    pattern = f"({'|'.join(map(re.escape, separators))})"

    for doc in docs:
        text = doc.get("content")
        if not isinstance(text, str):
            continue

        doc_lang = doc.get("language", "en").lower().strip()
        if doc_lang != language:
            continue

        parts = re.split(pattern, text)

        # Preserve separators by merging every pair of text + separator.
        sentences = []
        for i in range(0, len(parts), 2):
            seg = parts[i]
            sep = parts[i + 1] if i + 1 < len(parts) else ""
            combined = f"{seg}{sep}"
            if combined:
                sentences.append(combined)

        current = []
        current_len = 0
        chunk_index = 0

        for sentence in sentences:
            sent_len = len(sentence)
            if current and current_len + sent_len > chunk_size:
                chunk_text = "".join(current).strip()
                chunk_metadata = doc.copy()
                chunk_metadata.pop("content", None)
                chunk_metadata["chunk_index"] = chunk_index
                chunks.append({"page_content": chunk_text, "metadata": chunk_metadata})
                chunk_index += 1

                if chunk_overlap > 0:
                    overlap_text = "".join(current)[-chunk_overlap:]
                    current = [overlap_text] if overlap_text else []
                    current_len = len(overlap_text)
                else:
                    current = []
                    current_len = 0

            current.append(sentence)
            current_len += sent_len

        if current:
            chunk_text = "".join(current).strip()
            chunk_metadata = doc.copy()
            chunk_metadata.pop("content", None)
            chunk_metadata["chunk_index"] = chunk_index
            chunks.append({"page_content": chunk_text, "metadata": chunk_metadata})

    return chunks