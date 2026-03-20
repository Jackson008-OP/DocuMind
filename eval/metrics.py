import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_dataset(path: str = "eval/dataset.json") -> list:
    """
    Loads the evaluation dataset from JSON file.
    """
    with open(path, "r") as f:
        data = json.load(f)
    return data["eval_dataset"]


def check_retrieval(result: dict, expected_source: str,
                    expected_page: int) -> bool:
    """
    Checks if the correct source was retrieved.
    Returns True if the expected document and page
    appears in the top results.
    """
    if result["status"] != "success":
        return False

    for source in result["sources"]:
        source_filename = source.get("filename", "")
        source_page     = source.get("page", 0)

        # Check if the right document was found
        if expected_source in source_filename:
            # Allow 1 page margin — chunking can shift page numbers
            if abs(int(source_page) - expected_page) <= 1:
                return True

    return False


def check_answer_quality(answer: str, expected_keywords: list) -> dict:
    """
    Checks how many expected keywords appear in the answer.
    Returns a score between 0.0 and 1.0

    Example: expected = ["python", "pandas", "numpy"]
             answer contains "python" and "pandas"
             score = 2/3 = 0.67
    """
    answer_lower = answer.lower()
    found        = []
    missing      = []

    for keyword in expected_keywords:
        if keyword.lower() in answer_lower:
            found.append(keyword)
        else:
            missing.append(keyword)

    score = len(found) / len(expected_keywords) if expected_keywords else 0.0

    return {
        "score":   score,
        "found":   found,
        "missing": missing
    }


def run_evaluation(pipeline, dataset: list) -> dict:
    """
    Runs the full evaluation against all questions.
    Returns a complete results dict with per-question
    scores and overall summary metrics.
    """
    results          = []
    retrieval_scores = []
    quality_scores   = []

    print(f"\nRunning evaluation on {len(dataset)} questions...\n")

    for item in dataset:
        question         = item["question"]
        expected_keywords = item["expected_keywords"]
        expected_source  = item["expected_source"]
        expected_page    = item["expected_page"]

        print(f"Q{item['id']}: {question}")

        # Run through DocuMind pipeline
        result = pipeline.ask(question)

        # Check retrieval
        retrieval_correct = check_retrieval(
            result, expected_source, expected_page
        )

        # Check answer quality
        quality = check_answer_quality(
            result["answer"], expected_keywords
        )

        retrieval_scores.append(1 if retrieval_correct else 0)
        quality_scores.append(quality["score"])

        # Store per-question result
        results.append({
            "id":                item["id"],
            "question":          question,
            "status":            result["status"],
            "retrieval_correct": retrieval_correct,
            "quality_score":     quality["score"],
            "keywords_found":    quality["found"],
            "keywords_missing":  quality["missing"],
            "answer_preview":    result["answer"][:120]
        })

        # Print per-question summary
        retrieval_icon = "✓" if retrieval_correct else "✗"
        print(f"   Retrieval : {retrieval_icon}")
        print(f"   Quality   : {quality['score']:.0%} "
              f"({len(quality['found'])}/{len(expected_keywords)} keywords)")
        if quality["missing"]:
            print(f"   Missing   : {quality['missing']}")
        print()

    # Calculate overall scores
    retrieval_precision = sum(retrieval_scores) / len(retrieval_scores)
    avg_quality         = sum(quality_scores)   / len(quality_scores)

    summary = {
        "total_questions":    len(dataset),
        "retrieval_precision": retrieval_precision,
        "avg_quality_score":  avg_quality,
        "results":            results
    }

    return summary


def print_report(summary: dict, chunk_size: int = 512) -> None:
    """
    Prints a clean evaluation report.
    """
    print("\n" + "="*55)
    print(f"  DOCUMIND EVALUATION REPORT (chunk_size={chunk_size})")
    print("="*55)
    print(f"  Total questions    : {summary['total_questions']}")
    print(f"  Retrieval precision: "
          f"{summary['retrieval_precision']:.0%} "
          f"({int(summary['retrieval_precision'] * summary['total_questions'])}"
          f"/{summary['total_questions']} correct)")
    print(f"  Answer quality     : {summary['avg_quality_score']:.0%}")
    print("="*55)

    # Grade
    precision = summary["retrieval_precision"]
    if precision >= 0.8:
        grade = "Excellent"
    elif precision >= 0.6:
        grade = "Good"
    elif precision >= 0.4:
        grade = "Fair"
    else:
        grade = "Needs improvement"

    print(f"  Grade              : {grade}")
    print("="*55)


# Quick test when run directly
if __name__ == "__main__":
    from generation.pipeline import DocuMindPipeline

    # Load pipeline and dataset
    pipeline = DocuMindPipeline()
    dataset  = load_dataset()

    # Run evaluation with default chunk size (512)
    summary = run_evaluation(pipeline, dataset)
    print_report(summary, chunk_size=512)