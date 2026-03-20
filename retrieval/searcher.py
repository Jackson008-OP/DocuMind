import logging
from embeddings.embedder    import Embedder
from embeddings.vectorstore import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Searcher:
    """
    Searches ChromaDB for chunks relevant to a query.
    Uses vector similarity — finds meaning, not just keywords.
    """

    def __init__(self, embedder: Embedder, store: VectorStore,
                 top_k: int = 5, min_score: float = 0.17):
        """
        embedder:  the embedding model (to convert query to vector)
        store:     the ChromaDB vector store
        top_k:     how many results to retrieve (we'll rerank these)
        min_score: minimum similarity score to be considered relevant
                   0.0 = no filter, 1.0 = perfect match only
                   0.15 works well for most document types
        """
        self.embedder  = embedder
        self.store     = store
        self.top_k     = top_k
        self.min_score = min_score

    def search(self, query: str) -> list:
        """
        Main search function.
        Takes a plain English question, returns relevant chunks.

        Each result is a dict with:
          text      → the chunk content
          metadata  → source, page, citation etc.
          score     → similarity score (0 to 1, higher is better)
          citation  → human readable source reference
        """
        if not query.strip():
            logger.warning("Empty query received")
            return []

        # Check if database has anything
        if self.store.count() == 0:
            logger.warning("Vector store is empty — no documents indexed yet")
            return []

        # Convert query text into a vector
        logger.info(f"Searching for: '{query}'")
        query_vector = self.embedder.embed_text(query)

        # Search ChromaDB for similar vectors
        results = self.store.search(query_vector, top_k=self.top_k)

        # Filter out results below minimum confidence
        filtered = [r for r in results if r["score"] >= self.min_score]

        if not filtered:
            logger.info("No results above confidence threshold")
            return []

        logger.info(f"Found {len(filtered)} relevant chunks")
        return filtered

    def search_with_context(self, query: str) -> dict:
        """
        Extended search that returns results AND a status message.
        This is what the generation layer will call.
        """
        results = self.search(query)

        if not results:
            return {
                "results":  [],
                "status":   "no_results",
                "message":  "I could not find relevant information in the loaded documents."
            }

        return {
            "results":  results,
            "status":   "success",
            "message":  f"Found {len(results)} relevant sections"
        }


# Quick test when run directly
if __name__ == "__main__":
    embedder = Embedder()
    store    = VectorStore()
    searcher = Searcher(embedder, store)

    # Test queries
    test_queries = [
        "what is this document about?",
        "what technologies are used?",
        "what are the future enhancements?",
        "what is the weather like on mars?"   # should return low/no results
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query: {query}")
        response = searcher.search_with_context(query)
        print(f"Status: {response['status']}")

        for i, result in enumerate(response["results"]):
            print(f"\n  Result {i+1}:")
            print(f"  Score   : {result['score']:.3f}")
            print(f"  Citation: {result['citation']}")
            print(f"  Text    : {result['text'][:120]}...")