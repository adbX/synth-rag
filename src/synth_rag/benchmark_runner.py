#!/usr/bin/env python3
"""
Benchmark RAG system performance on RAGBench dataset using hybrid search.
Retrieves contexts and generates responses, saving results for later evaluation.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from fastembed import TextEmbedding, SparseTextEmbedding
from langchain_openai import ChatOpenAI
from qdrant_client import models
from tqdm import tqdm

from synth_rag.settings import (
    get_qdrant_client,
    get_api_settings,
    ensure_benchmark_logs_dir,
    RAGBENCH_DATASETS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RAGBench benchmark using hybrid search"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=RAGBENCH_DATASETS,
        default="emanual",
        help="RAGBench sub-dataset to benchmark",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "validation", "test"],
        default="test",
        help="Dataset split to use for benchmarking",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Qdrant collection name (default: ragbench_{dataset})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for response generation",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of contexts to retrieve per query",
    )
    parser.add_argument(
        "--prefetch-limit",
        type=int,
        default=50,
        help="Number of results to prefetch for reranking",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Maximum number of examples to process (for testing)",
    )
    return parser.parse_args()


class HybridSearchRetriever:
    """Hybrid search retriever using dense + sparse embeddings (no ColPali)."""

    def __init__(self, collection_name: str, top_k: int = 5, prefetch_limit: int = 50):
        self.collection_name = collection_name
        self.top_k = top_k
        self.prefetch_limit = prefetch_limit
        self.client = get_qdrant_client()

        # Load embedding models
        print("Loading FastEmbed models...")
        self.dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        self.sparse_model = SparseTextEmbedding("Qdrant/bm25")

    def retrieve(self, query: str) -> dict:
        """
        Retrieve relevant contexts for a query using hybrid search.

        Returns:
            dict with retrieved_contexts, scores, and metadata
        """
        # Generate embeddings
        dense_embedding = list(self.dense_model.embed([query]))[0]
        sparse_embedding = list(self.sparse_model.embed([query]))[0]

        # Perform hybrid search
        start_time = datetime.now()

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=dense_embedding.tolist(),
            query_filter=None,
            limit=self.top_k,
            prefetch=[
                # Prefetch using dense embeddings
                models.Prefetch(
                    query=dense_embedding.tolist(),
                    limit=self.prefetch_limit,
                    using="dense",
                ),
                # Prefetch using sparse embeddings
                models.Prefetch(
                    query=sparse_embedding.as_object(),
                    limit=self.prefetch_limit,
                    using="sparse",
                ),
            ],
            using="dense",  # Final scoring using dense
            with_payload=True,
        )

        query_time = (datetime.now() - start_time).total_seconds()

        # Extract results
        contexts = []
        scores = []
        metadata = []

        for point in response.points:
            contexts.append(point.payload.get("chunk_text", ""))
            scores.append(point.score)
            metadata.append(
                {
                    "example_id": point.payload.get("example_id"),
                    "document_idx": point.payload.get("document_idx"),
                    "chunk_idx": point.payload.get("chunk_idx"),
                    "point_id": point.id,
                }
            )

        return {
            "contexts": contexts,
            "scores": scores,
            "metadata": metadata,
            "query_time_seconds": query_time,
            "num_retrieved": len(contexts),
        }


def generate_response(question: str, contexts: list[str], model: str) -> dict:
    """
    Generate a response using retrieved contexts and LLM.

    Returns:
        dict with generated_response and metadata
    """
    # Setup OpenAI
    api_settings = get_api_settings()
    os.environ["OPENAI_API_KEY"] = api_settings.openai_key

    llm = ChatOpenAI(model=model, temperature=0)

    # Create prompt
    context_text = "\n\n".join([f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)])

    prompt = f"""Answer the following question based on the provided context passages.

Context:
{context_text}

Question: {question}

Answer: Provide a comprehensive answer based only on the information in the context passages above."""

    # Generate response
    start_time = datetime.now()
    response = llm.invoke(prompt)
    generation_time = (datetime.now() - start_time).total_seconds()

    return {
        "generated_response": response.content,
        "generation_time_seconds": generation_time,
        "model": model,
        "prompt_length": len(prompt),
    }


def run_benchmark(
    dataset_name: str,
    split: str,
    collection_name: str,
    model: str,
    top_k: int,
    prefetch_limit: int,
    max_examples: int | None = None,
) -> Path:
    """
    Run benchmark on RAGBench dataset.

    Returns:
        Path to the saved results file
    """
    # Use default collection name if not provided
    if collection_name is None:
        collection_name = f"ragbench_{dataset_name}"

    # Load dataset
    print(f"Loading RAGBench dataset: {dataset_name} ({split} split)")
    dataset = load_dataset("rungalileo/ragbench", dataset_name, split=split)
    print(f"✓ Loaded {len(dataset)} examples")

    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))
        print(f"✓ Using first {len(dataset)} examples for testing")

    # Initialize retriever
    retriever = HybridSearchRetriever(
        collection_name=collection_name, top_k=top_k, prefetch_limit=prefetch_limit
    )

    # Prepare output directory and files
    logs_dir = ensure_benchmark_logs_dir(dataset_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save run configuration
    config = {
        "dataset_name": dataset_name,
        "split": split,
        "collection_name": collection_name,
        "model": model,
        "top_k": top_k,
        "prefetch_limit": prefetch_limit,
        "num_examples": len(dataset),
        "timestamp": timestamp,
    }

    config_path = logs_dir / f"{timestamp}_run_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\n✓ Saved run config to: {config_path}")

    # Process each example
    results_path = logs_dir / f"{timestamp}_raw_results.jsonl"
    
    print(f"\nRunning benchmark on {len(dataset)} examples...")
    print(f"Results will be saved to: {results_path}\n")

    with open(results_path, "w") as f:
        for example in tqdm(dataset, desc="Processing examples"):
            example_id = example.get("id")
            question = example.get("question")
            ground_truth_documents = example.get("documents", [])
            ground_truth_response = example.get("response")

            # Retrieve contexts
            retrieval_result = retriever.retrieve(question)

            # Generate response
            generation_result = generate_response(
                question=question,
                contexts=retrieval_result["contexts"],
                model=model,
            )

            # Combine results
            result = {
                "example_id": example_id,
                "question": question,
                "ground_truth_documents": ground_truth_documents,
                "ground_truth_response": ground_truth_response,
                "retrieved_contexts": retrieval_result["contexts"],
                "retrieval_scores": retrieval_result["scores"],
                "retrieval_metadata": retrieval_result["metadata"],
                "query_time_seconds": retrieval_result["query_time_seconds"],
                "generated_response": generation_result["generated_response"],
                "generation_time_seconds": generation_result["generation_time_seconds"],
                "total_time_seconds": retrieval_result["query_time_seconds"]
                + generation_result["generation_time_seconds"],
                # Include ground truth metrics from dataset
                "ground_truth_adherence_score": example.get("adherence_score"),
                "ground_truth_relevance_score": example.get("relevance_score"),
                "ground_truth_utilization_score": example.get("utilization_score"),
                "ground_truth_completeness_score": example.get("completeness_score"),
            }

            # Write result as JSONL
            f.write(json.dumps(result) + "\n")

    print(f"\n✅ Benchmark complete!")
    print(f"✓ Results saved to: {results_path}")
    print(f"✓ Config saved to: {config_path}")

    return results_path


def main():
    args = parse_args()

    results_path = run_benchmark(
        dataset_name=args.dataset,
        split=args.split,
        collection_name=args.collection,
        model=args.model,
        top_k=args.top_k,
        prefetch_limit=args.prefetch_limit,
        max_examples=args.max_examples,
    )

    print(f"\nNext step: Run metrics evaluation on {results_path}")
    print(f"  uv run python -m synth_rag.benchmark_metrics --results-file {results_path}")


if __name__ == "__main__":
    main()

