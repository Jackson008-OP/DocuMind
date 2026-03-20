import re


def clean_text(text: str) -> str:
    """
    Takes raw text from a document page and removes noise.
    Returns clean text ready for chunking and embedding.
    """

    # Remove page numbers like: "Page 1", "- 1 -", "1 of 9"
    text = re.sub(r'\bpage\s+\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'-\s*\d+\s*-', '', text)
    text = re.sub(r'\d+\s+of\s+\d+', '', text, flags=re.IGNORECASE)

    # Remove URLs (they add noise, not meaning)
    text = re.sub(r'http[s]?://\S+', '', text)

    # Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)

    # Replace multiple blank lines with a single blank line
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Replace multiple spaces with a single space
    text = re.sub(r' {2,}', ' ', text)

    # Remove lines that are just whitespace or single characters
    # (these are usually leftover header/footer artifacts)
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        stripped = line.strip()
        # Keep the line only if it has more than 2 characters
        if len(stripped) > 2:
            cleaned_lines.append(stripped)

    # Join lines back together
    text = '\n'.join(cleaned_lines)

    # Final strip of leading/trailing whitespace
    return text.strip()


def clean_documents(documents: list) -> list:
    """
    Takes a list of document objects (from loader.py)
    and cleans the text content of each one.
    Returns the same list with cleaned text.
    """
    cleaned = []

    for doc in documents:
        # Clean the page_content of each document
        original_length = len(doc.page_content)
        doc.page_content = clean_text(doc.page_content)
        cleaned_length = len(doc.page_content)

        # Skip documents that became empty after cleaning
        if not doc.page_content:
            continue

        cleaned.append(doc)

    print(f"Cleaned {len(cleaned)} documents")
    return cleaned


# Quick test when run directly
if __name__ == "__main__":
    # Import loader to get real documents
    from ingest.loader import load_documents

    docs = load_documents()
    cleaned_docs = clean_documents(docs)

    print(f"\nBefore cleaning: {len(docs)} docs")
    print(f"After cleaning:  {len(cleaned_docs)} docs")
    print(f"\nSample cleaned text:\n{cleaned_docs[0].page_content[:300]}")