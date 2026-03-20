import re
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_citation_numbers(answer: str) -> list:
    """
    Finds all citation numbers in Mistral's answer.
    Example: "Python is used [1] and NumPy too [2]"
             returns [1, 2]
    """
    # Find all patterns like [1], [2], [12] etc.
    matches = re.findall(r'\[(\d+)\]', answer)

    # Convert to integers and remove duplicates
    # while preserving order of first appearance
    seen = set()
    numbers = []
    for m in matches:
        n = int(m)
        if n not in seen:
            seen.add(n)
            numbers.append(n)

    return numbers


def build_sources_list(citation_numbers: list, chunks: list) -> list:
    """
    Maps citation numbers back to real source information.

    citation_numbers: list of ints like [1, 2]
    chunks:           the chunks that were sent to Mistral
                      (same order = same numbering)

    Returns a list of source dicts with full citation info.
    """
    sources = []

    for num in citation_numbers:
        # Citation numbers start at 1 but list index starts at 0
        index = num - 1

        if 0 <= index < len(chunks):
            chunk = chunks[index]
            sources.append({
                "number":   num,
                "citation": chunk.get("citation", f"Source {num}"),
                "filename": chunk.get("metadata", {}).get("filename", "Unknown"),
                "page":     chunk.get("metadata", {}).get("page_number", "?"),
                "preview":  chunk.get("text", "")[:100] + "..."
            })
        else:
            # Citation number out of range — still include it
            sources.append({
                "number":   num,
                "citation": f"Source {num} (not found)",
                "filename": "Unknown",
                "page":     "?",
                "preview":  ""
            })

    return sources


def format_final_response(answer: str, chunks: list) -> dict:
    """
    Takes Mistral's raw answer and the chunks that were used,
    and returns a clean structured response ready for the UI.

    Returns a dict with:
      answer      → Mistral's answer text (unchanged)
      sources     → list of source dicts with full citation info
      has_sources → True if any citations were found
      source_text → formatted string of sources for display
    """
    # Extract which sources Mistral cited
    citation_numbers = extract_citation_numbers(answer)

    # Map numbers to real source info
    sources = build_sources_list(citation_numbers, chunks)

    # Build a readable sources string for the UI
    source_lines = []
    for source in sources:
        source_lines.append(
            f"[{source['number']}] {source['citation']}"
        )
    source_text = "\n".join(source_lines)

    return {
        "answer":      answer,
        "sources":     sources,
        "has_sources": len(sources) > 0,
        "source_text": source_text
    }


# Quick test when run directly
if __name__ == "__main__":

    # Simulate Mistral's answer with citations
    test_answer = ("The main technologies used are Python, Pandas, "
                   "and NumPy [1]. Future enhancements include "
                   "real-time data integration [2].")

    test_chunks = [
        {
            "text":     "Python, Pandas, NumPy and Scikit-learn are used.",
            "citation": "Report.pdf — Page 8, Chunk 11",
            "metadata": {"filename": "Report.pdf", "page_number": 8}
        },
        {
            "text":     "Future enhancements include real-time integration.",
            "citation": "Report.pdf — Page 9, Chunk 12",
            "metadata": {"filename": "Report.pdf", "page_number": 9}
        }
    ]

    print(f"Mistral's answer:\n{test_answer}")
    print(f"\nCitation numbers found: {extract_citation_numbers(test_answer)}")

    result = format_final_response(test_answer, test_chunks)

    print(f"\n=== Formatted Response ===")
    print(f"Answer:\n{result['answer']}")
    print(f"\nSources:")
    print(result['source_text'])
    print(f"\nHas sources: {result['has_sources']}")
    print(f"\nDetailed sources:")
    for s in result['sources']:
        print(f"  [{s['number']}] {s['citation']}")
        print(f"       Page: {s['page']}")
        print(f"       Preview: {s['preview']}")