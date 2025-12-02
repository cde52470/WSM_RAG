from typing import Any, Dict, List, Optional


def _validate_chunk_params(chunk_size: int, chunk_overlap: int) -> int:
    """Validate chunk configuration and return the step size."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size to avoid an infinite loop.")
    return chunk_size - chunk_overlap


def chunk_documents(
    docs: List[Dict[str, Any]],
    language: Optional[str],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Dict[str, Any]]:
    """
    Split documents into overlapping character chunks.

    Only documents whose language matches the provided `language` are chunked.
    When `language` is None or empty, all documents are processed.
    """
    step = _validate_chunk_params(chunk_size, chunk_overlap)
    chunks: List[Dict[str, Any]] = []

    for doc_index, doc in enumerate(docs):
        text = doc.get("content")
        lang = doc.get("language")

        if not isinstance(text, str) or not text.strip():
            # Skip empty or malformed documents to avoid silent failures later.
            continue
        if language and lang != language:
            continue

        text = text.strip()
        text_len = len(text)
        start_index = 0
        chunk_count = 0

        while start_index < text_len:
            end_index = min(start_index + chunk_size, text_len)

            chunk_metadata = {k: v for k, v in doc.items() if k != "content"}
            chunk_metadata.update(
                {
                    "chunk_index": chunk_count,
                    "doc_index": doc_index,
                    "char_start": start_index,
                    "char_end": end_index,
                }
            )

            chunks.append(
                {
                    "page_content": text[start_index:end_index],
                    "metadata": chunk_metadata,
                }
            )

            start_index += step
            chunk_count += 1

    return chunks
