import hashlib
from datetime import datetime


def enrich_metadata(chunks: list) -> list:
    """
    Adds extra metadata to every chunk so we can:
    - Display clean source citations
    - Track which chunks are already indexed
    - Identify every chunk with a unique ID
    """

    for i, chunk in enumerate(chunks):

        # Get the existing metadata (source + page from LangChain)
        source = chunk.metadata.get("source", "unknown")
        page   = chunk.metadata.get("page", 0)

        # Clean up the filename for display
        # "docs\\Report.pdf" → "Report.pdf"
        import os
        filename = os.path.basename(source)

        # Create a unique ID for this chunk using a hash
        # This means the same chunk always gets the same ID
        # even if you re-run the indexing
        hash_input = f"{filename}-page{page}-chunk{i}-{chunk.page_content[:50]}"
        chunk_id = hashlib.md5(hash_input.encode()).hexdigest()[:12]

        # Add all the new metadata fields
        chunk.metadata.update({
            "filename":        filename,
            "page_number":     page + 1,        # humans count from 1, not 0
            "chunk_index":     i,
            "chunk_id":        chunk_id,
            "indexed_at":      datetime.now().isoformat(),
            "content_length":  len(chunk.page_content),

            # This is the human-readable citation string
            "citation": f"{filename} — Page {page + 1}, Chunk {i + 1}"
        })

    print(f"Enriched metadata for {len(chunks)} chunks")
    return chunks


# Quick test when run directly
if __name__ == "__main__":
    from ingest.loader import load_documents
    from ingest.cleaner import clean_documents
    from ingest.chunker import chunk_documents

    docs   = load_documents()
    cleaned = clean_documents(docs)
    chunks  = chunk_documents(cleaned)
    chunks  = enrich_metadata(chunks)

    # Show the full metadata of the first chunk
    print(f"\n--- Full Metadata for Chunk 1 ---")
    for key, value in chunks[0].metadata.items():
        print(f"  {key}: {value}")

    print(f"\n--- Full Metadata for Chunk 2 ---")
    for key, value in chunks[1].metadata.items():
        print(f"  {key}: {value}")