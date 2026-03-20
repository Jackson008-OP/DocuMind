import logging
from embeddings.embedder       import Embedder
from embeddings.vectorstore    import VectorStore
from retrieval.searcher        import Searcher
from retrieval.reranker        import Reranker
from retrieval.router          import route_query
from generation.prompt         import (build_prompt,
                                       build_chitchat_prompt,
                                       build_no_results_prompt)
from generation.context_manager import prepare_context
from generation.llm            import generate_response
from generation.citations      import format_final_response

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocuMindPipeline:
    """
    The complete DocuMind pipeline.
    Loads all models once, then answers any number of questions.

    Usage:
        pipeline = DocuMindPipeline()
        result   = pipeline.ask("what technologies are used?")
        print(result["answer"])
        print(result["source_text"])
    """

    def __init__(self):
        logger.info("Initialising DocuMind pipeline...")

        # Load all models once — reused for every question
        self.embedder = Embedder()
        self.store    = VectorStore()
        self.searcher = Searcher(self.embedder, self.store)
        self.reranker = Reranker()

        logger.info("DocuMind pipeline ready!")

    def ask(self, query: str) -> dict:
        """
        Main entry point — takes a question, returns a full response.

        Returns a dict with:
          answer      → Mistral's answer text
          sources     → list of source dicts
          source_text → formatted sources string
          has_sources → True if citations were found
          status      → 'success', 'no_results', or 'chitchat'
          query_type  → 'specific', 'summary', or 'chitchat'
        """
        logger.info(f"Processing query: '{query}'")

        # Step 1 — Route the query
        # Decides: chitchat / summary / specific / no_results
        routed = route_query(query, self.searcher, self.reranker)

        # Step 2 — Handle chitchat separately
        if routed["status"] == "chitchat":
            prompt   = build_chitchat_prompt(query)
            answer   = generate_response(prompt)
            return {
                "answer":      answer,
                "sources":     [],
                "source_text": "",
                "has_sources": False,
                "status":      "chitchat",
                "query_type":  "chitchat"
            }

        # Step 3 — Handle no results
        if routed["status"] == "no_results":
            prompt   = build_no_results_prompt(query)
            answer   = generate_response(prompt)
            return {
                "answer":      answer,
                "sources":     [],
                "source_text": "",
                "has_sources": False,
                "status":      "no_results",
                "query_type":  routed["query_type"]
            }

        # Step 4 — Prepare context (trim if too long)
        context  = prepare_context(routed["results"])
        chunks   = context["chunks"]

        if context["was_trimmed"]:
            logger.info(f"Context trimmed: {context['original_count']} "
                        f"→ {context['final_count']} chunks")

        # Step 5 — Build the prompt
        prompt = build_prompt(query, chunks)

        # Step 6 — Send to Mistral and get answer
        raw_answer = generate_response(prompt)

        # Step 7 — Parse citations and format response
        result = format_final_response(raw_answer, chunks)
        result["status"]     = "success"
        result["query_type"] = routed["query_type"]

        logger.info("Query answered successfully")
        return result


# Quick test — full end to end!
if __name__ == "__main__":
    pipeline = DocuMindPipeline()

    test_questions = [
        "hello!",
        "what technologies are used in this project?",
        "what are the future enhancements?",
        "what is the weather like on mars?"
    ]

    for question in test_questions:
        print(f"\n{'='*55}")
        print(f"Question: {question}")
        print(f"{'='*55}")

        result = pipeline.ask(question)

        print(f"Status    : {result['status']}")
        print(f"Type      : {result['query_type']}")
        print(f"\nAnswer:\n{result['answer']}")

        if result["has_sources"]:
            print(f"\nSources:\n{result['source_text']}")