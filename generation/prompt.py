def build_prompt(query: str, chunks: list) -> str:
    """
    Builds the full prompt to send to Mistral.

    query:  the user's question
    chunks: list of retrieved + reranked result dicts
            each has 'text' and 'citation' keys

    The prompt has 3 parts:
    1. System instruction — defines Mistral's role
    2. Context block — the retrieved chunks, numbered
    3. Question + answer instruction
    """

    # Part 1 — System instruction
    system = """You are DocuMind, an intelligent document assistant.
Your job is to answer questions based ONLY on the provided context sections below.

Rules you must follow:
- Only use information from the context sections provided
- Always cite your sources using [1], [2], [3] etc.
- If the context does not contain enough information, say so honestly
- Do not use your general training knowledge to fill gaps
- Keep answers clear, concise and well structured
- If multiple context sections are relevant, synthesise them into one answer"""

    # Part 2 — Context block
    # Number each chunk so Mistral can reference them
    context_lines = []
    for i, chunk in enumerate(chunks):
        citation = chunk.get("citation", f"Source {i+1}")
        text     = chunk.get("text", "").strip()
        context_lines.append(f"[{i+1}] {citation}\n{text}")

    context_block = "\n\n".join(context_lines)

    # Part 3 — The actual question
    question_block = f"""Based on the context sections above, please answer this question:
{query}

Remember to cite which context sections [1], [2] etc. your answer comes from."""

    # Combine all three parts into the final prompt
    full_prompt = f"""{system}

---CONTEXT START---
{context_block}
---CONTEXT END---

{question_block}"""

    return full_prompt


def build_chitchat_prompt(query: str) -> str:
    """
    A simpler prompt for conversational messages
    that don't need document retrieval.
    """
    return f"""You are DocuMind, a helpful document assistant.
The user has sent a conversational message. Respond briefly and helpfully.
Let them know you can answer questions about their uploaded documents.

User message: {query}"""


def build_no_results_prompt(query: str) -> str:
    """
    Prompt used when no relevant chunks were found.
    Tells Mistral to respond honestly instead of guessing.
    """
    return f"""You are DocuMind, a helpful document assistant.
The user asked: "{query}"

No relevant information was found in the uploaded documents for this question.
Politely explain that you could not find this information in the loaded documents,
and suggest they try rephrasing their question or uploading more relevant documents.
Do not attempt to answer from general knowledge."""


# Quick test when run directly
if __name__ == "__main__":
    # Simulate some chunks
    test_chunks = [
        {
            "text": "Python, Pandas, NumPy and Scikit-learn are the main technologies used.",
            "citation": "Report.pdf — Page 8, Chunk 11"
        },
        {
            "text": "Future enhancements include real-time data integration and district-level analysis.",
            "citation": "Report.pdf — Page 9, Chunk 12"
        }
    ]

    prompt = build_prompt("what technologies are used?", test_chunks)
    print("=== FULL PROMPT ===")
    print(prompt)
    print(f"\n=== PROMPT LENGTH: {len(prompt)} characters ===")