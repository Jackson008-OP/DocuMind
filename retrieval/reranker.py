from sentence_transformers import CrossEncoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Reranker:
    """
    Takes search results and re-scores them for better accuracy.
    Uses a CrossEncoder model that reads query + chunk together
    instead of comparing them as separate vectors.

    CrossEncoder is slower but more accurate than vector similarity.
    We only run it on top 5 candidates — never the whole database.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        First run downloads the model (~80MB). Cached after that.
        """
        logger.info(f"Loading reranker model: {model_name}")
        self.model = CrossEncoder(model_name)
        logger.info("Reranker model loaded successfully")

    def rerank(self, query: str, results: list, top_n: int = 3) -> list:
        """
        Re-scores and re-orders search results.

        query:   the original user question
        results: list of results from searcher.py
        top_n:   how many top results to keep after reranking

        Returns the top_n most relevant results in order.
        """
        if not results:
            return []

        # Build pairs of [query, chunk_text] for the cross encoder
        # The model reads both together to judge relevance
        pairs = [[query, result["text"]] for result in results]

        # Score each pair — higher score = more relevant
        scores = self.model.predict(pairs)

        # Attach reranker scores to results
        for i, result in enumerate(results):
            result["rerank_score"] = float(scores[i])

        # Sort by reranker score (highest first)
        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

        # Keep only top_n results
        top_results = reranked[:top_n]

        logger.info(f"Reranked {len(results)} → kept top {len(top_results)}")
        return top_results


# Quick test when run directly
if __name__ == "__main__":
    from embeddings.embedder    import Embedder
    from embeddings.vectorstore import VectorStore
    from retrieval.searcher     import Searcher

    embedder = Embedder()
    store    = VectorStore()
    searcher = Searcher(embedder, store)
    reranker = Reranker()

    query = "what technologies are used in this project?"

    print(f"\nQuery: {query}")
    print("\n--- Before Reranking ---")
    results = searcher.search(query)
    for i, r in enumerate(results):
        print(f"  {i+1}. Score:{r['score']:.3f} | {r['citation']}")
        print(f"     {r['text'][:80]}...")

    print("\n--- After Reranking ---")
    reranked = reranker.rerank(query, results, top_n=3)
    for i, r in enumerate(reranked):
        print(f"  {i+1}. Rerank:{r['rerank_score']:.3f} | {r['citation']}")
        print(f"     {r['text'][:80]}...")