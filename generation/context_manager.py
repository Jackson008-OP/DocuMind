import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mistral 7B can handle roughly 4000 characters of context safely
# We keep a buffer so the answer has room too
MAX_CONTEXT_CHARS = 3500


def count_chars(chunks: list) -> int:
    """
    Counts total characters across all chunk texts.
    """
    return sum(len(chunk.get("text", "")) for chunk in chunks)


def trim_chunks(chunks: list, max_chars: int = MAX_CONTEXT_CHARS) -> list:
    """
    Trims the chunk list to fit within max_chars.

    Strategy: keep chunks from the top (most relevant first)
    and remove from the bottom until we fit.
    Chunks are already sorted best-first by the reranker
    so we always keep the most important ones.
    """
    if count_chars(chunks) <= max_chars:
        # Already fits — no trimming needed
        return chunks

    logger.info(f"Context too long — trimming chunks to fit {max_chars} chars")

    kept   = []
    total  = 0

    for chunk in chunks:
        chunk_len = len(chunk.get("text", ""))

        # Check if adding this chunk would exceed the limit
        if total + chunk_len <= max_chars:
            kept.append(chunk)
            total += chunk_len
        else:
            # Try to fit a truncated version of this chunk
            remaining = max_chars - total
            if remaining > 100:
                # Truncate the chunk text to fit remaining space
                truncated = dict(chunk)
                truncated["text"] = chunk["text"][:remaining] + "..."
                kept.append(truncated)
            break

    logger.info(f"Kept {len(kept)} of {len(chunks)} chunks "
                f"({total} chars)")
    return kept


def prepare_context(chunks: list) -> dict:
    """
    Main function — takes raw reranked chunks and prepares
    them for the prompt builder.

    Returns a dict with:
      chunks        → trimmed list ready for prompt
      total_chars   → character count
      was_trimmed   → True if we had to cut anything
      original_count → how many chunks before trimming
    """
    original_count = len(chunks)
    original_chars = count_chars(chunks)

    trimmed = trim_chunks(chunks)

    return {
        "chunks":         trimmed,
        "total_chars":    count_chars(trimmed),
        "was_trimmed":    len(trimmed) < original_count,
        "original_count": original_count,
        "final_count":    len(trimmed)
    }


# Quick test when run directly
if __name__ == "__main__":
    # Simulate chunks of different sizes
    test_chunks = [
        {"text": "A" * 1000, "citation": "Doc1 — Page 1"},
        {"text": "B" * 1000, "citation": "Doc1 — Page 2"},
        {"text": "C" * 1000, "citation": "Doc1 — Page 3"},
        {"text": "D" * 1000, "citation": "Doc1 — Page 4"},
        {"text": "E" * 1000, "citation": "Doc1 — Page 5"},
    ]

    print(f"Input  : {len(test_chunks)} chunks, "
          f"{count_chars(test_chunks)} total chars")

    result = prepare_context(test_chunks)

    print(f"Output : {result['final_count']} chunks, "
          f"{result['total_chars']} total chars")
    print(f"Trimmed: {result['was_trimmed']}")
    print(f"\nTest 2 — small input that fits without trimming:")

    small_chunks = [
        {"text": "Short text here", "citation": "Doc — Page 1"},
        {"text": "Another short text", "citation": "Doc — Page 2"},
    ]

    result2 = prepare_context(small_chunks)
    print(f"Output : {result2['final_count']} chunks, "
          f"{result2['total_chars']} chars")
    print(f"Trimmed: {result2['was_trimmed']}")