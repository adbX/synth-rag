#!/usr/bin/env python3
"""
Compute evaluation metrics for RAGBench benchmark results.
Uses ragas, trulens, and custom evaluation functions.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from synth_rag.settings import get_api_settings

# Set API keys from environment
api_settings = get_api_settings()
os.environ["OPENAI_API_KEY"] = api_settings.openai_key

# Add ragbench code to path
RAGBENCH_CODE_DIR = Path(__file__).resolve().parents[3] / "prompts" / "ragbench-main"
sys.path.insert(0, str(RAGBENCH_CODE_DIR))

# Import after path is set
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

# Import TruLens (new API)
try:
    from trulens.providers.openai import OpenAI as TrulensOpenAI
except ImportError:
    try:
        # Fallback to old API if new one not available
        from trulens_eval.feedback.provider.openai import OpenAI as TrulensOpenAI
    except ImportError:
        print("Warning: Could not import TruLens OpenAI provider")
        TrulensOpenAI = None


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
    Compute RAGAS metrics (faithfulness, answer_relevancy).

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

    # Configure RAGAS with OpenAI LLM
    evaluator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # Run RAGAS evaluation with configured LLM
    ragas_result = evaluate(
        ragas_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=evaluator_llm,
    )

    # Convert to DataFrame
    ragas_df = ragas_result.to_pandas()

    # Add metrics back to results
    for i, result in enumerate(results):
        result["ragas_faithfulness"] = ragas_df.iloc[i]["faithfulness"]
        result["ragas_answer_relevancy"] = ragas_df.iloc[i]["answer_relevancy"]

    print("✓ RAGAS metrics computed")
    
    # Handle potential NaN values in output
    faithfulness_mean = ragas_df['faithfulness'].mean()
    answer_relevancy_mean = ragas_df['answer_relevancy'].mean()
    
    if pd.notna(faithfulness_mean):
        print(f"  - Mean faithfulness: {faithfulness_mean:.3f}")
    else:
        print(f"  - Mean faithfulness: NaN (check for errors)")
        
    if pd.notna(answer_relevancy_mean):
        print(f"  - Mean answer_relevancy: {answer_relevancy_mean:.3f}")
    else:
        print(f"  - Mean answer_relevancy: NaN (check for errors)")

    return results


def compute_trulens_metrics(results: list[dict]) -> list[dict]:
    """
    Compute TruLens metrics (groundedness, context_relevance) synchronously.

    Returns:
        Updated results with trulens metrics
    """
    print("\n" + "=" * 80)
    print("Computing TruLens Metrics")
    print("=" * 80)

    if TrulensOpenAI is None:
        print("✗ TruLens OpenAI provider not available, skipping TruLens metrics")
        return results

    # Initialize TruLens provider
    trulens_provider = TrulensOpenAI()

    print(f"Evaluating {len(results)} examples with TruLens...")
    print("Note: Processing sequentially (non-concurrent)")

    # Process each result sequentially
    for result in tqdm(results, desc="TruLens evaluation"):
        contexts = result["retrieved_contexts"]
        question = result["question"]
        response = result["generated_response"]

        try:
            # Compute groundedness (faithfulness)
            groundedness_score = trulens_provider.groundedness_measure_with_cot_reasons(
                source=contexts,
                statement=response
            )
            result["trulens_groundedness"] = groundedness_score[0] if groundedness_score else None

            # Compute context relevance
            context_relevance_score = trulens_provider.context_relevance_with_cot_reasons(
                question=question,
                context=contexts
            )
            result["trulens_context_relevance"] = context_relevance_score[0] if context_relevance_score else None

        except Exception as e:
            print(f"\n  Warning: Error evaluating example {result['example_id']}: {e}")
            result["trulens_groundedness"] = None
            result["trulens_context_relevance"] = None

    print("\n✓ TruLens metrics computed")
    
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

    # RAGAS metrics (if available, filtering out NaN values)
    ragas_faithfulness = [
        r["ragas_faithfulness"] for r in results 
        if "ragas_faithfulness" in r and not pd.isna(r["ragas_faithfulness"])
    ]
    ragas_answer_relevancy = [
        r["ragas_answer_relevancy"] for r in results 
        if "ragas_answer_relevancy" in r and not pd.isna(r["ragas_answer_relevancy"])
    ]

    if ragas_faithfulness:
        metrics["ragas_faithfulness_mean"] = np.mean(ragas_faithfulness)
        metrics["ragas_faithfulness_std"] = np.std(ragas_faithfulness)
    else:
        metrics["ragas_faithfulness_mean"] = None
        metrics["ragas_faithfulness_std"] = None
    
    if ragas_answer_relevancy:
        metrics["ragas_answer_relevancy_mean"] = np.mean(ragas_answer_relevancy)
        metrics["ragas_answer_relevancy_std"] = np.std(ragas_answer_relevancy)
    else:
        metrics["ragas_answer_relevancy_mean"] = None
        metrics["ragas_answer_relevancy_std"] = None

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
    metrics["hallucination_auroc_ragas"] = None
    if ragas_faithfulness:
        ground_truth_adherence = [
            r["ground_truth_adherence_score"]
            for r in results
            if r.get("ground_truth_adherence_score") is not None
        ]
        
        if ground_truth_adherence and len(ground_truth_adherence) == len(ragas_faithfulness):
            # Convert adherence to hallucination (invert)
            trues_hallucination = ~np.array(ground_truth_adherence, dtype=bool)
            preds_hallucination = 1 - np.array(ragas_faithfulness, dtype=float)
            
            try:
                metrics["hallucination_auroc_ragas"] = auroc(
                    trues_hallucination.tolist(), preds_hallucination.tolist()
                )
            except Exception as e:
                print(f"  Warning: Could not compute hallucination AUROC: {e}")
                metrics["hallucination_auroc_ragas"] = None

    # Relevance RMSE
    metrics["relevance_rmse_ragas"] = None
    if ragas_answer_relevancy:
        ground_truth_relevance = [
            r["ground_truth_relevance_score"]
            for r in results
            if r.get("ground_truth_relevance_score") is not None
        ]
        
        if ground_truth_relevance and len(ground_truth_relevance) == len(ragas_answer_relevancy):
            try:
                metrics["relevance_rmse_ragas"] = rmse(
                    ground_truth_relevance, ragas_answer_relevancy
                )
            except Exception as e:
                print(f"  Warning: Could not compute relevance RMSE: {e}")
                metrics["relevance_rmse_ragas"] = None

    # Print summary
    print(f"✓ Processed {metrics['num_examples']} examples")
    print(f"✓ Mean query time: {metrics['mean_query_time']:.3f}s")
    print(f"✓ Mean generation time: {metrics['mean_generation_time']:.3f}s")
    print(f"✓ Mean total time: {metrics['mean_total_time']:.3f}s")

    if "ragas_faithfulness_mean" in metrics and metrics['ragas_faithfulness_mean'] is not None:
        print(f"✓ RAGAS faithfulness: {metrics['ragas_faithfulness_mean']:.3f} ± {metrics['ragas_faithfulness_std']:.3f}")
    if "ragas_answer_relevancy_mean" in metrics and metrics['ragas_answer_relevancy_mean'] is not None:
        print(f"✓ RAGAS context relevancy: {metrics['ragas_answer_relevancy_mean']:.3f} ± {metrics['ragas_answer_relevancy_std']:.3f}")
    if "trulens_groundedness_mean" in metrics and metrics['trulens_groundedness_mean'] is not None:
        print(f"✓ TruLens groundedness: {metrics['trulens_groundedness_mean']:.3f} ± {metrics['trulens_groundedness_std']:.3f}")
    if "trulens_context_relevance_mean" in metrics and metrics['trulens_context_relevance_mean'] is not None:
        print(f"✓ TruLens context relevance: {metrics['trulens_context_relevance_mean']:.3f} ± {metrics['trulens_context_relevance_std']:.3f}")
    if "hallucination_auroc_ragas" in metrics and metrics['hallucination_auroc_ragas'] is not None:
        print(f"✓ Hallucination AUROC (RAGAS): {metrics['hallucination_auroc_ragas']:.3f}")
    if "relevance_rmse_ragas" in metrics and metrics['relevance_rmse_ragas'] is not None:
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


def main():
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
            results = compute_trulens_metrics(results)
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


if __name__ == "__main__":
    main()

