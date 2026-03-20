import logging
import json
import gc
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def clear_chroma():
    """
    Safely closes ChromaDB and deletes the folder on Windows.
    """
    import shutil
    import time

    chroma_path = Path("chroma_db")
    if chroma_path.exists():
        # Force garbage collection to release file handles
        gc.collect()
        time.sleep(2)
        try:
            shutil.rmtree(chroma_path)
            print("Cleared existing index")
        except PermissionError:
            # If still locked, try deleting file by file
            for f in chroma_path.rglob("*"):
                try:
                    if f.is_file():
                        f.unlink()
                except:
                    pass
            try:
                shutil.rmtree(chroma_path, ignore_errors=True)
                print("Cleared existing index")
            except:
                print("Warning: could not fully clear index — continuing anyway")


def run_comparison():
    """
    Runs evaluation with two different chunk sizes
    and compares the results side by side.
    """
    from embeddings.index_docs  import index_documents
    from eval.metrics           import load_dataset, run_evaluation, print_report

    dataset     = load_dataset()
    all_results = {}

    for chunk_size in [512, 256]:
        print(f"\n{'='*55}")
        print(f"  TESTING CHUNK SIZE: {chunk_size}")
        print(f"{'='*55}")

        # Clear old index safely
        clear_chroma()

        # Re-index with this chunk size
        print(f"Re-indexing with chunk_size={chunk_size}...")
        index_documents(chunk_size=chunk_size, force_reindex=True)

        # Import pipeline fresh each time
        # Re-importing forces new ChromaDB connection
        import importlib
        import generation.pipeline as pipeline_module
        importlib.reload(pipeline_module)
        pipeline = pipeline_module.DocuMindPipeline()

        # Run evaluation
        summary = run_evaluation(pipeline, dataset)
        print_report(summary, chunk_size=chunk_size)

        # Store results
        all_results[chunk_size] = summary

        # Explicitly delete pipeline to release ChromaDB file handles
        del pipeline
        gc.collect()

    # Print final comparison table
    print(f"\n{'='*55}")
    print(f"  CHUNK SIZE COMPARISON TABLE")
    print(f"{'='*55}")
    print(f"  {'Metric':<25} {'512':>10} {'256':>10}")
    print(f"  {'-'*45}")

    r512 = all_results[512]
    r256 = all_results[256]

    print(f"  {'Retrieval precision':<25} "
          f"{r512['retrieval_precision']:>9.0%} "
          f"{r256['retrieval_precision']:>9.0%}")

    print(f"  {'Answer quality':<25} "
          f"{r512['avg_quality_score']:>9.0%} "
          f"{r256['avg_quality_score']:>9.0%}")

    print(f"  {'Questions correct':<25} "
          f"{int(r512['retrieval_precision']*10):>10} "
          f"{int(r256['retrieval_precision']*10):>10}")

    print(f"{'='*55}")

    # Recommendation
    if r512["retrieval_precision"] >= r256["retrieval_precision"]:
        winner = 512
    else:
        winner = 256

    print(f"\n  Recommended chunk size: {winner}")
    print(f"  Reason: Higher retrieval precision on this document")
    print(f"{'='*55}")

    # Save results
    save_results(all_results)

    # Re-index with winning chunk size at the end
    print(f"\nRe-indexing with winning chunk size ({winner})...")
    clear_chroma()
    index_documents(chunk_size=winner, force_reindex=True)
    print("Done! DocuMind is ready with optimal settings.")


def save_results(all_results: dict) -> None:
    """
    Saves evaluation results to eval/results/ folder.
    """
    Path("eval/results").mkdir(exist_ok=True)

    output = {}
    for chunk_size, summary in all_results.items():
        output[str(chunk_size)] = {
            "chunk_size":          chunk_size,
            "retrieval_precision": summary["retrieval_precision"],
            "avg_quality_score":   summary["avg_quality_score"],
            "total_questions":     summary["total_questions"]
        }

    with open("eval/results/comparison.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to eval/results/comparison.json")


if __name__ == "__main__":
    run_comparison()