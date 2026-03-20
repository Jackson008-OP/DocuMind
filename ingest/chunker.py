from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_documents(documents: list, chunk_size: int = 512,
                    chunk_overlap: int = 50) -> list:
    """
    Splits document pages into smaller chunks for embedding.

    chunk_size    = max characters per chunk (we'll test 256 vs 512)
    chunk_overlap = how many characters to repeat between chunks
                    so context isn't lost at the edges
    """

    # RecursiveCharacterTextSplitter tries to split in this order:
    # 1. At paragraph breaks (\n\n)
    # 2. At line breaks (\n)
    # 3. At spaces
    # 4. At individual characters (last resort)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,          # measure size by character count
        separators=["\n\n", "\n", " ", ""]
    )

    # Split all documents into chunks
    chunks = splitter.split_documents(documents)

    print(f"Split {len(documents)} documents into {len(chunks)} chunks")
    print(f"Settings: chunk_size={chunk_size}, overlap={chunk_overlap}")

    return chunks


def chunk_documents_small(documents: list) -> list:
    """
    Convenience function for chunk size 256.
    We'll use this during evaluation to compare results.
    """
    return chunk_documents(documents, chunk_size=256, chunk_overlap=30)


def chunk_documents_large(documents: list) -> list:
    """
    Convenience function for chunk size 512.
    We'll use this during evaluation to compare results.
    """
    return chunk_documents(documents, chunk_size=512, chunk_overlap=50)


# Quick test when run directly
if __name__ == "__main__":
    from ingest.loader import load_documents
    from ingest.cleaner import clean_documents

    # Load and clean first
    docs = load_documents()
    cleaned = clean_documents(docs)

    # Now chunk with default size (512)
    chunks = chunk_documents(cleaned)

    # Show a sample chunk so we can see what it looks like
    print(f"\n--- Sample Chunk ---")
    print(f"Content:\n{chunks[0].page_content}")
    print(f"\nMetadata: {chunks[0].metadata}")