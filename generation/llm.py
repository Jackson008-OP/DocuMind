import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ollama runs locally on this address — no internet needed
OLLAMA_URL   = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mistral"


def check_ollama_running() -> bool:
    """
    Checks if Ollama is running on your machine.
    Returns True if it's running, False if not.
    """
    try:
        response = requests.get(
            "http://localhost:11434/api/tags",
            timeout=3
        )
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def generate_response(prompt: str,
                      model: str = DEFAULT_MODEL,
                      temperature: float = 0.1) -> str:
    """
    Sends a prompt to Mistral via Ollama and returns the response.

    prompt:      the full prompt built by prompt.py
    model:       which Ollama model to use (default: mistral)
    temperature: how creative the response is
                 0.0 = very focused and factual (best for RAG)
                 1.0 = more creative and varied
                 We use 0.1 to keep answers grounded and precise
    """

    # Check Ollama is running before trying
    if not check_ollama_running():
        error_msg = ("Ollama is not running. "
                     "Please start it with: ollama serve")
        logger.error(error_msg)
        return error_msg

    # Build the request payload
    payload = {
        "model":  model,
        "prompt": prompt,
        "stream": False,       # get full response at once
        "options": {
            "temperature": temperature,
            "num_predict": 512,   # max tokens in response
            "top_p":       0.9,
        }
    }

    try:
        logger.info(f"Sending prompt to {model} via Ollama...")

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=120    # 2 minutes max — Mistral can be slow on CPU
        )

        response.raise_for_status()

        # Parse the JSON response
        result = response.json()
        answer = result.get("response", "").strip()

        logger.info("Response received successfully")
        return answer

    except requests.exceptions.Timeout:
        logger.error("Ollama request timed out")
        return "Request timed out. Please try again."

    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to Ollama")
        return "Could not connect to Ollama. Is it running?"

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return f"An error occurred: {str(e)}"


# Quick test when run directly
if __name__ == "__main__":

    print("Checking Ollama status...")
    if not check_ollama_running():
        print("Ollama is NOT running!")
        print("Start it by running: ollama serve")
        exit(1)

    print("Ollama is running!")
    print("\nSending test prompt to Mistral...")

    test_prompt = """You are DocuMind, a document assistant.
Answer this question in one sentence using only the context below.

Context: Python, Pandas, NumPy and Scikit-learn are the main technologies used.

Question: What technologies are used?
Cite your answer as [1]."""

    response = generate_response(test_prompt)

    print(f"\n=== Mistral's Response ===")
    print(response)