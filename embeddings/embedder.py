from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Embedder:
    """
    Loads the embedding model once and reuses it.
    Loading a model is slow — we only want to do it one time.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        model_name: the HuggingFace model to use.
        all-MiniLM-L6-v2 is small, fast, and high quality.
        It produces 384-dimensional vectors.
        First run downloads it (~90MB). After that it's cached.
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully")

    def embed_text(self, text: str) -> list:
        """
        Converts a single string into a vector (list of 384 numbers).
        Used for embedding search queries at runtime.
        """
        vector = self.model.encode(text, convert_to_numpy=True)
        return vector.tolist()

    def embed_texts(self, texts: list) -> list:
        """
        Converts a list of strings into a list of vectors.
        More efficient than calling embed_text() in a loop
        because it batches the work.
        Used for embedding all chunks during indexing.
        """
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,    # shows a progress bar for large batches
            batch_size=32              # process 32 chunks at a time
        )
        return vectors.tolist()


# Quick test when run directly
if __name__ == "__main__":
    embedder = Embedder()

    # Test with two similar sentences and one different one
    test_texts = [
        "The cat sat on the mat",
        "A cat was sitting on a mat",   # similar meaning
        "The stock market crashed today" # very different meaning
    ]

    print("\nEmbedding 3 test sentences...")
    vectors = embedder.embed_texts(test_texts)

    print(f"\nVector dimensions: {len(vectors[0])} numbers per chunk")
    print(f"First 5 numbers of sentence 1: {vectors[0][:5]}")
    print(f"First 5 numbers of sentence 2: {vectors[1][:5]}")
    print("\nNotice sentences 1 and 2 have similar numbers")
    print("because they have similar meaning — that's the magic!")