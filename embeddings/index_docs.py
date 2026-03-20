import logging
from pathlib import Path

from ingest.loader      import load_documents
from ingest.cleaner     import clean_documents
from ingest.chunker     import chunk_documents
from ingest.metadata    import enrich_metadata
from embeddings.embedder     import Embedder
from embeddings.vectorstore  import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_indexed_files(store: VectorStore) -> set:
    """
    Returns a set of filenames already stored in ChromaDB.
    We use this to skip files that are already indexed.
    """
    return set(store.get_all_documents())


def index_documents(docs_dir: str = "docs",
                    chunk_size: int = 512,
                    force_reindex: bool = False) -> dict:
    """
    Master pipeline — runs the full ingestion flow.

    docs_dir:      folder to scan for documents
    chunk_size:    size of each chunk (512 default)
    force_reindex: if True, re-processes all files even
                   if they're already indexed

    Returns a summary dictionary with stats.
    """

    logger.info("=== DocuMind Indexing Pipeline Started ===")

    # Step 1 — Set up embedder and vector store
    embedder = Embedder()
    store    = VectorStore()

    # Step 2 — Find which files are already indexed
    already_indexed = get_indexed_files(store)
    if already_indexed:
        logger.info(f"Already indexed: {already_indexed}")

    # Step 3 — Load all documents from docs/ folder
    all_docs = load_documents(docs_dir)
    if not all_docs:
        logger.warning("No documents found in docs/ folder")
        return {"status": "no_documents", "indexed": 0, "skipped": 0}

    # Step 4 — Filter out already-indexed files
    # unless force_reindex is True
    if not force_reindex:
        new_docs = []
        skipped  = set()

        for doc in all_docs:
            import os
            filename = os.path.basename(doc.metadata.get("source", ""))
            if filename in already_indexed:
                skipped.add(filename)
            else:
                new_docs.append(doc)

        if skipped:
            logger.info(f"Skipping already-indexed files: {skipped}")

        if not new_docs:
            logger.info("All documents already indexed — nothing to do!")
            return {
                "status":   "already_indexed",
                "indexed":  0,
                "skipped":  len(skipped),
                "total_chunks": store.count()
            }
    else:
        new_docs = all_docs
        skipped  = set()

    # Step 5 — Run the full pipeline on new documents only
    logger.info(f"Processing {len(new_docs)} new document pages...")

    cleaned = clean_documents(new_docs)
    chunks  = chunk_documents(cleaned, chunk_size=chunk_size)
    chunks  = enrich_metadata(chunks)

    # Step 6 — Store in ChromaDB
    store.add_chunks(chunks, embedder)

    # Step 7 — Build summary
    summary = {
        "status":       "success",
        "new_pages":    len(new_docs),
        "new_chunks":   len(chunks),
        "skipped":      len(skipped),
        "total_chunks": store.count(),
        "documents":    store.get_all_documents()
    }

    logger.info("=== Indexing Complete ===")
    logger.info(f"New chunks added : {len(chunks)}")
    logger.info(f"Total in database: {store.count()}")
    logger.info(f"All documents    : {store.get_all_documents()}")

    return summary


# Run directly to index all documents in docs/ folder
if __name__ == "__main__":
    summary = index_documents()

    print("\n=== Indexing Summary ===")
    for key, value in summary.items():
        print(f"  {key}: {value}")