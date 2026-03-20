import logging
from retrieval.searcher  import Searcher
from retrieval.reranker  import Reranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SUMMARY_KEYWORDS = [
    "summarise", "summarize", "summary", "overview",
    "everything", "all about", "tell me about",
    "what does this document",
    "explain the document", "describe the document"
]

CHITCHAT_KEYWORDS = [
    "hello", "hi there", "hey there", "good morning",
    "good evening", "thanks", "thank you", "goodbye",
    "bye", "how are you", "what can you do",
    "who are you", "what are you"
]


def classify_query(query: str) -> str:
    """
    Classifies the query into one of three types:
    'chitchat'  → greeting or conversational
    'summary'   → user wants a broad overview
    'specific'  → user wants a specific fact
    """
    query_lower = query.lower().strip()
    word_count  = len(query_lower.split())

    # Check summary first — always, regardless of length
    for keyword in SUMMARY_KEYWORDS:
        if keyword in query_lower:
            return "summary"

    # Chitchat only applies to SHORT queries (5 words or less)
    # This prevents real questions like "what technologies are
    # used in this project?" from being misclassified
    if word_count <= 5:
        for keyword in CHITCHAT_KEYWORDS:
            if keyword in query_lower:
                return "chitchat"

    # Default — specific factual question
    return "specific"


def route_query(query: str, searcher: Searcher,
                reranker: Reranker) -> dict:
    """
    Main routing function — decides how to handle the query
    and returns a structured response for the generation layer.

    Returns a dict with:
      query_type  → 'chitchat', 'summary', or 'specific'
      results     → list of relevant chunks (empty for chitchat)
      status      → 'success', 'no_results', or 'chitchat'
      message     → human readable status description
    """
    query_type = classify_query(query)
    logger.info(f"Query type: '{query_type}' for: '{query}'")

    # Handle chitchat — no retrieval needed
    if query_type == "chitchat":
        return {
            "query_type": "chitchat",
            "results":    [],
            "status":     "chitchat",
            "message":    "Conversational query — no retrieval needed"
        }

    # For summary queries — retrieve more chunks (top 5)
    # For specific queries — retrieve fewer but more precise (top 3)
    top_k = 5 if query_type == "summary" else 3
    top_n = 3 if query_type == "summary" else 2

    # Run searcher
    search_response = searcher.search_with_context(query)

    # If nothing found — return honest no_results
    if search_response["status"] == "no_results":
        return {
            "query_type": query_type,
            "results":    [],
            "status":     "no_results",
            "message":    "I could not find relevant information in the loaded documents."
        }

    # Rerank the results for better precision
    reranked = reranker.rerank(
        query,
        search_response["results"],
        top_n=top_n
    )

    return {
        "query_type": query_type,
        "results":    reranked,
        "status":     "success",
        "message":    f"Found {len(reranked)} relevant sections"
    }


# Quick test when run directly
if __name__ == "__main__":
    from embeddings.embedder    import Embedder
    from embeddings.vectorstore import VectorStore

    embedder = Embedder()
    store    = VectorStore()
    searcher = Searcher(embedder, store)
    reranker = Reranker()

    test_queries = [
        "hello there!",
        "summarise this document",
        "what technologies are used in this project?",
        "what is the weather like on mars?",
    ]

    for query in test_queries:
        print(f"\n{'='*50}")
        print(f"Query     : {query}")
        response = route_query(query, searcher, reranker)
        print(f"Type      : {response['query_type']}")
        print(f"Status    : {response['status']}")
        print(f"Chunks    : {len(response['results'])}")
        if response["results"]:
            print(f"Top result: {response['results'][0]['citation']}")