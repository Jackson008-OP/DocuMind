import chromadb
from chromadb.config import Settings
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages the ChromaDB vector database.
    All vectors are saved to disk in chroma_db/ folder.
    Data persists between sessions — nothing is lost on restart.
    """

    def __init__(self, persist_dir: str = "chroma_db"):
        """
        persist_dir: folder where ChromaDB saves its data.
        Created automatically if it doesn't exist.
        """
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        # A "collection" is like a table in a database
        # We use one collection for all our document chunks
        self.collection = self.client.get_or_create_collection(
            name="documind",
            metadata={"hnsw:space": "cosine"}  # cosine similarity for text
        )

        logger.info(f"VectorStore ready — {self.collection.count()} chunks stored")

    def add_chunks(self, chunks: list, embedder) -> None:
        """
        Adds document chunks + their vectors to ChromaDB.
        Skips chunks that are already stored (by chunk_id)
        so re-running never creates duplicates.
        """
        if not chunks:
            logger.warning("No chunks to add")
            return

        # Get all the text content from chunks
        texts = [chunk.page_content for chunk in chunks]

        # Generate vectors for all chunks at once
        logger.info(f"Embedding {len(chunks)} chunks...")
        vectors = embedder.embed_texts(texts)

        # Prepare metadata and IDs for ChromaDB
        ids       = [chunk.metadata["chunk_id"]  for chunk in chunks]
        metadatas = [chunk.metadata               for chunk in chunks]

        # Add everything to ChromaDB in one batch
        # upsert = insert if new, update if exists (no duplicates)
        self.collection.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas
        )

        logger.info(f"Stored {len(chunks)} chunks — "
                    f"total in database: {self.collection.count()}")

    def search(self, query_vector: list, top_k: int = 5) -> list:
        """
        Finds the top_k most similar chunks to a query vector.
        Returns a list of results with text + metadata + similarity score.
        """
        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, self.collection.count()),
            include=["documents", "metadatas", "distances"]
        )

        # Format results into clean dictionaries
        formatted = []
        for i in range(len(results["documents"][0])):
            formatted.append({
                "text":      results["documents"][0][i],
                "metadata":  results["metadatas"][0][i],
                "score":     1 - results["distances"][0][i],  # convert to similarity
                "citation":  results["metadatas"][0][i].get("citation", "Unknown source")
            })

        return formatted

    def get_all_documents(self) -> list:
        """
        Returns a list of unique document filenames stored in the database.
        Used by the UI to show which documents are loaded.
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.get(include=["metadatas"])
        filenames = set()
        for metadata in results["metadatas"]:
            filenames.add(metadata.get("filename", "unknown"))

        return sorted(list(filenames))

    def delete_document(self, filename: str) -> None:
        """
        Removes all chunks belonging to a specific document.
        Used by the document manager in the UI.
        """
        results = self.collection.get(include=["metadatas"])
        ids_to_delete = []

        for i, metadata in enumerate(results["metadatas"]):
            if metadata.get("filename") == filename:
                ids_to_delete.append(results["ids"][i])

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
            logger.info(f"Deleted {len(ids_to_delete)} chunks for {filename}")
        else:
            logger.warning(f"No chunks found for {filename}")

    def count(self) -> int:
        return self.collection.count()


# Quick test when run directly
if __name__ == "__main__":
    from ingest.loader   import load_documents
    from ingest.cleaner  import clean_documents
    from ingest.chunker  import chunk_documents
    from ingest.metadata import enrich_metadata
    from embeddings.embedder import Embedder

    # Full pipeline test
    docs    = load_documents()
    cleaned = clean_documents(docs)
    chunks  = chunk_documents(cleaned)
    chunks  = enrich_metadata(chunks)

    # Set up embedder and vector store
    embedder = Embedder()
    store    = VectorStore()

    # Store all chunks
    store.add_chunks(chunks, embedder)

    # Test a search
    print("\n--- Testing Search ---")
    query = "what is this document about?"
    query_vector = embedder.embed_text(query)
    results = store.search(query_vector, top_k=3)

    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"  Score:    {result['score']:.3f}")
        print(f"  Citation: {result['citation']}")
        print(f"  Text:     {result['text'][:100]}...")

    print(f"\nDocuments in database: {store.get_all_documents()}")