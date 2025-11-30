#!/usr/bin/env python3
"""
Compute evaluation metrics for RAGBench benchmark results.
Uses ragas, trulens, and custom evaluation functions.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import roc_auc_score

# Add ragbench code to path
RAGBENCH_CODE_DIR = Path(__file__).resolve().parents[3] / "prompts" / "ragbench-main"
sys.path.insert(0, str(RAGBENCH_CODE_DIR))

from ragas import evaluate
from ragas.metrics import faithfulness, context_relevancy

# Import ragbench evaluation utilities
try:
    from ragbench.trulens_async import AsyncTrulensOpenAI
except ImportError:
    print("Warning: Could not import AsyncTrulensOpenAI from ragbench")
    AsyncTrulensOpenAI = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute evaluation metrics for RAGBench benchmark results"
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        required=True,
        help="Path to JSONL results file from benchmark_runner",
    )
    parser.add_argument(
        "--skip-ragas",
        action="store_true",
        help="Skip ragas metrics computation",
    )
    parser.add_argument(
        "--skip-trulens",
        action="store_true",
        help="Skip trulens metrics computation",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=10,
        help="Max concurrent requests for trulens evaluation",
    )
    return parser.parse_args()


def load_results(results_file: Path) -> list[dict]:
    """Load results from JSONL file."""
    results = []
    with open(results_file, "r") as f:
        for line in f:
            results.append(json.loads(line))
    return results


def rmse(trues: list[float], preds: list[float]) -> float:
    """Calculate Root Mean Squared Error (RMSE)."""
    if len(trues) != len(preds):
        return None

    trues = np.array(trues)
    preds = np.array(preds, dtype=float)

    # Ignore Nulls in predictions
    eval_idx = ~np.isnan(preds)
    trues = trues[eval_idx]
    preds = preds[eval_idx]

    if len(trues) == 0:
        return None

    return np.sqrt(np.mean((preds - trues) ** 2))


def auroc(trues: list[bool], preds: list[float]) -> float:
    """Calculate Area Under Receiver Operator Characteristic Curve (AUROC)."""
    trues = np.array(trues)
    preds = np.array(preds, dtype=float)
    
    eval_idx = ~np.isnan(preds)
    
    if eval_idx.sum() == 0:
        return None
    
    return roc_auc_score(trues[eval_idx], preds[eval_idx])


def compute_ragas_metrics(results: list[dict]) -> list[dict]:
    """
    Compute RAGAS metrics (faithfulness, context_relevancy).

    Returns:
        Updated results with ragas metrics
    """
    print("\n" + "=" * 80)
    print("Computing RAGAS Metrics")
    print("=" * 80)

    # Prepare data for RAGAS
    ragas_data = {
        "question": [],
        "contexts": [],
        "answer": [],
    }

    for result in results:
        ragas_data["question"].append(result["question"])
        ragas_data["contexts"].append(result["retrieved_contexts"])
        ragas_data["answer"].append(result["generated_response"])

    # Create dataset
    ragas_dataset = Dataset.from_dict(ragas_data)

    print(f"Evaluating {len(ragas_dataset)} examples with RAGAS...")

    # Run RAGAS evaluation
    ragas_result = evaluate(
        ragas_dataset,
        metrics=[faithfulness, context_relevancy],
    )

    # Convert to DataFrame
    ragas_df = ragas_result.to_pandas()

    # Add metrics back to results
    for i, result in enumerate(results):
        result["ragas_faithfulness"] = ragas_df.iloc[i]["faithfulness"]
        result["ragas_context_relevancy"] = ragas_df.iloc[i]["context_relevancy"]

    print("✓ RAGAS metrics computed")
    print(f"  - Mean faithfulness: {ragas_df['faithfulness'].mean():.3f}")
    print(f"  - Mean context_relevancy: {ragas_df['context_relevancy'].mean():.3f}")

    return results


async def compute_trulens_metrics_async(
    results: list[dict], max_concurrent: int = 10
) -> list[dict]:
    """
    Compute TruLens metrics (groundedness, context_relevance) asynchronously.

    Returns:
        Updated results with trulens metrics
    """
    print("\n" + "=" * 80)
    print("Computing TruLens Metrics (Async)")
    print("=" * 80)

    if AsyncTrulensOpenAI is None:
        print("✗ AsyncTrulensOpenAI not available, skipping TruLens metrics")
        return results

    # Initialize TruLens
    async_trulens = AsyncTrulensOpenAI()

    # Prepare data
    ids = [result["example_id"] for result in results]
    contexts = [result["retrieved_contexts"] for result in results]
    questions = [result["question"] for result in results]
    responses = [result["generated_response"] for result in results]

    print(f"Evaluating {len(results)} examples with TruLens...")

    # Run async evaluation
    trulens_results = await async_trulens.annotate(
        ids=ids,
        contexts=contexts,
        questions=questions,
        responses=responses,
        metrics=["groundedness", "context_relevance"],
        max_concurrent=max_concurrent,
    )

    # Add metrics back to results
    for i, result in enumerate(results):
        annotation = trulens_results[i]
        
        result["trulens_groundedness"] = (
            annotation.groundedness.value if annotation.groundedness else None
        )
        result["trulens_context_relevance"] = (
            annotation.context_relevance.value if annotation.context_relevance else None
        )

    print("✓ TruLens metrics computed")
    
    groundedness_values = [
        r["trulens_groundedness"] for r in results if r.get("trulens_groundedness") is not None
    ]
    context_rel_values = [
        r["trulens_context_relevance"] for r in results if r.get("trulens_context_relevance") is not None
    ]
    
    if groundedness_values:
        print(f"  - Mean groundedness: {np.mean(groundedness_values):.3f}")
    if context_rel_values:
        print(f"  - Mean context_relevance: {np.mean(context_rel_values):.3f}")

    return results


def compute_aggregate_metrics(results: list[dict]) -> dict:
    """
    Compute aggregate metrics and comparisons with ground truth.

    Returns:
        Dictionary of aggregate metrics
    """
    print("\n" + "=" * 80)
    print("Computing Aggregate Metrics")
    print("=" * 80)

    metrics = {}

    # Basic statistics
    metrics["num_examples"] = len(results)
    metrics["mean_query_time"] = np.mean([r["query_time_seconds"] for r in results])
    metrics["mean_generation_time"] = np.mean(
        [r["generation_time_seconds"] for r in results]
    )
    metrics["mean_total_time"] = np.mean([r["total_time_seconds"] for r in results])

    # RAGAS metrics (if available)
    ragas_faithfulness = [
        r["ragas_faithfulness"] for r in results if "ragas_faithfulness" in r
    ]
    ragas_context_relevancy = [
        r["ragas_context_relevancy"] for r in results if "ragas_context_relevancy" in r
    ]

    if ragas_faithfulness:
        metrics["ragas_faithfulness_mean"] = np.mean(ragas_faithfulness)
        metrics["ragas_faithfulness_std"] = np.std(ragas_faithfulness)
    
    if ragas_context_relevancy:
        metrics["ragas_context_relevancy_mean"] = np.mean(ragas_context_relevancy)
        metrics["ragas_context_relevancy_std"] = np.std(ragas_context_relevancy)

    # TruLens metrics (if available)
    trulens_groundedness = [
        r["trulens_groundedness"]
        for r in results
        if r.get("trulens_groundedness") is not None
    ]
    trulens_context_relevance = [
        r["trulens_context_relevance"]
        for r in results
        if r.get("trulens_context_relevance") is not None
    ]

    if trulens_groundedness:
        metrics["trulens_groundedness_mean"] = np.mean(trulens_groundedness)
        metrics["trulens_groundedness_std"] = np.std(trulens_groundedness)
    
    if trulens_context_relevance:
        metrics["trulens_context_relevance_mean"] = np.mean(trulens_context_relevance)
        metrics["trulens_context_relevance_std"] = np.std(trulens_context_relevance)

    # Hallucination detection (using RAGAS faithfulness as proxy for adherence)
    if ragas_faithfulness:
        ground_truth_adherence = [
            r["ground_truth_adherence_score"]
            for r in results
            if r.get("ground_truth_adherence_score") is not None
        ]
        
        if ground_truth_adherence:
            # Convert adherence to hallucination (invert)
            trues_hallucination = ~np.array(ground_truth_adherence, dtype=bool)
            preds_hallucination = 1 - np.array(ragas_faithfulness, dtype=float)
            
            try:
                metrics["hallucination_auroc_ragas"] = auroc(
                    trues_hallucination.tolist(), preds_hallucination.tolist()
                )
            except Exception as e:
                print(f"  Warning: Could not compute hallucination AUROC: {e}")

    # Relevance RMSE
    if ragas_context_relevancy:
        ground_truth_relevance = [
            r["ground_truth_relevance_score"]
            for r in results
            if r.get("ground_truth_relevance_score") is not None
        ]
        
        if ground_truth_relevance:
            try:
                metrics["relevance_rmse_ragas"] = rmse(
                    ground_truth_relevance, ragas_context_relevancy
                )
            except Exception as e:
                print(f"  Warning: Could not compute relevance RMSE: {e}")

    # Print summary
    print(f"✓ Processed {metrics['num_examples']} examples")
    print(f"✓ Mean query time: {metrics['mean_query_time']:.3f}s")
    print(f"✓ Mean generation time: {metrics['mean_generation_time']:.3f}s")
    print(f"✓ Mean total time: {metrics['mean_total_time']:.3f}s")

    if "ragas_faithfulness_mean" in metrics:
        print(f"✓ RAGAS faithfulness: {metrics['ragas_faithfulness_mean']:.3f} ± {metrics['ragas_faithfulness_std']:.3f}")
    if "ragas_context_relevancy_mean" in metrics:
        print(f"✓ RAGAS context relevancy: {metrics['ragas_context_relevancy_mean']:.3f} ± {metrics['ragas_context_relevancy_std']:.3f}")
    if "trulens_groundedness_mean" in metrics:
        print(f"✓ TruLens groundedness: {metrics['trulens_groundedness_mean']:.3f} ± {metrics['trulens_groundedness_std']:.3f}")
    if "trulens_context_relevance_mean" in metrics:
        print(f"✓ TruLens context relevance: {metrics['trulens_context_relevance_mean']:.3f} ± {metrics['trulens_context_relevance_std']:.3f}")
    if "hallucination_auroc_ragas" in metrics:
        print(f"✓ Hallucination AUROC (RAGAS): {metrics['hallucination_auroc_ragas']:.3f}")
    if "relevance_rmse_ragas" in metrics:
        print(f"✓ Relevance RMSE (RAGAS): {metrics['relevance_rmse_ragas']:.3f}")

    return metrics


def save_results(
    results: list[dict],
    aggregate_metrics: dict,
    results_file: Path,
):
    """Save detailed results and aggregate metrics."""
    output_dir = results_file.parent
    base_name = results_file.stem.replace("_raw_results", "")

    # Save detailed results with metrics
    detailed_results_path = output_dir / f"{base_name}_detailed_results.jsonl"
    with open(detailed_results_path, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print(f"\n✓ Saved detailed results to: {detailed_results_path}")

    # Save aggregate metrics
    metrics_path = output_dir / f"{base_name}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(aggregate_metrics, f, indent=2)
    print(f"✓ Saved aggregate metrics to: {metrics_path}")

    # Save summary CSV
    summary_df = pd.DataFrame([aggregate_metrics])
    summary_path = output_dir / f"{base_name}_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Saved summary CSV to: {summary_path}")


async def main_async():
    args = parse_args()

    # Load results
    print(f"Loading results from: {args.results_file}")
    results = load_results(args.results_file)
    print(f"✓ Loaded {len(results)} results")

    # Compute RAGAS metrics
    if not args.skip_ragas:
        try:
            results = compute_ragas_metrics(results)
        except Exception as e:
            print(f"✗ Error computing RAGAS metrics: {e}")
            import traceback
            traceback.print_exc()

    # Compute TruLens metrics
    if not args.skip_trulens:
        try:
            results = await compute_trulens_metrics_async(
                results, max_concurrent=args.max_concurrent
            )
        except Exception as e:
            print(f"✗ Error computing TruLens metrics: {e}")
            import traceback
            traceback.print_exc()

    # Compute aggregate metrics
    aggregate_metrics = compute_aggregate_metrics(results)

    # Save results
    save_results(results, aggregate_metrics, args.results_file)

    print("\n" + "=" * 80)
    print("✅ Metrics computation complete!")
    print("=" * 80)


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

