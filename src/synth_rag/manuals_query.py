#!/usr/bin/env python3
"""
Query PDF manuals using hybrid search (dense + sparse + ColPali reranking).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import models

from synth_rag.settings import (
    get_qdrant_client,
    ensure_logs_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query PDF manuals with hybrid search")
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="midi_manuals",
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to return",
    )
    parser.add_argument(
        "--prefetch-limit",
        type=int,
        default=50,
        help="Number of results to prefetch for reranking",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device for ColPali model (mps, cuda:0, cpu)",
    )
    parser.add_argument(
        "--manual-filter",
        type=str,
        default=None,
        help="Filter by manual name (optional)",
    )
    return parser.parse_args()


def query_manuals(
    question: str,
    collection_name: str,
    top_k: int,
    prefetch_limit: int,
    device: str,
    manual_filter: str | None = None,
) -> dict:
    """
    Query manuals using hybrid search with ColPali reranking.
    
    Returns a dict with results and metadata.
    """
    client = get_qdrant_client()
    
    # Load models
    print("Loading ColPali model...")
    colpali_model = ColPali.from_pretrained(
        "vidore/colpali-v1.3",
        torch_dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    colpali_processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3")
    
    print("Loading FastEmbed models...")
    dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    sparse_model = SparseTextEmbedding("Qdrant/bm25")
    
    # Generate query embeddings
    print(f"Generating embeddings for: '{question}'")
    
    # Dense embedding
    dense_embedding = list(dense_model.embed([question]))[0]
    
    # Sparse embedding
    sparse_embedding = list(sparse_model.embed([question]))[0]
    
    # ColPali query embedding
    processed_query = colpali_processor.process_queries([question]).to(device)
    with torch.no_grad():
        query_embedding = colpali_model(**processed_query)[0]
    query_embedding_list = query_embedding.cpu().float().numpy().tolist()
    
    # Build filter if manual_filter is provided
    query_filter = None
    if manual_filter:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="manual_name",
                    match=models.MatchValue(value=manual_filter),
                )
            ]
        )
    
    # Perform hybrid search with prefetch and rerank
    print(f"Querying collection '{collection_name}'...")
    
    start_time = datetime.now()
    
    response = client.query_points(
        collection_name=collection_name,
        query=query_embedding_list,
        prefetch=[
            # Prefetch using dense embeddings
            models.Prefetch(
                query=dense_embedding.tolist(),
                limit=prefetch_limit,
                using="dense",
            ),
            # Prefetch using sparse embeddings
            models.Prefetch(
                query=sparse_embedding.as_object(),
                limit=prefetch_limit,
                using="sparse",
            ),
            # Prefetch using ColPali rows
            models.Prefetch(
                query=query_embedding_list,
                limit=prefetch_limit,
                using="colpali_rows",
            ),
            # Prefetch using ColPali columns
            models.Prefetch(
                query=query_embedding_list,
                limit=prefetch_limit,
                using="colpali_cols",
            ),
        ],
        limit=top_k,
        using="colpali_original",  # Final rerank with original ColPali vectors
        query_filter=query_filter,
        with_payload=True,
    )
    
    query_time = (datetime.now() - start_time).total_seconds()
    
    # Format results
    results = []
    for idx, point in enumerate(response.points):
        results.append({
            "rank": idx + 1,
            "score": point.score,
            "manual_name": point.payload.get("manual_name"),
            "page_num": point.payload.get("page_num"),
            "text_snippet": point.payload.get("text", "")[:500],
            "image_path": point.payload.get("image_path"),
            "point_id": point.id,
        })
    
    return {
        "question": question,
        "collection": collection_name,
        "top_k": top_k,
        "prefetch_limit": prefetch_limit,
        "manual_filter": manual_filter,
        "query_time_seconds": query_time,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }


def pretty_print_results(query_result: dict):
    """Pretty print query results."""
    print("\n" + "=" * 80)
    print(f"QUESTION: {query_result['question']}")
    print(f"Collection: {query_result['collection']}")
    print(f"Query time: {query_result['query_time_seconds']:.3f}s")
    print("=" * 80)
    
    if not query_result["results"]:
        print("\nNo results found.")
        return
    
    for result in query_result["results"]:
        print(f"\n[{result['rank']}] Score: {result['score']:.4f}")
        print(f"Manual: {result['manual_name']}")
        print(f"Page: {result['page_num']}")
        print(f"Image: {result['image_path']}")
        print(f"\nText snippet:")
        print(f"{result['text_snippet']}")
        print("-" * 80)


def save_query_log(query_result: dict) -> Path:
    """Save query results to logs directory."""
    logs_dir = ensure_logs_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"{timestamp}.json"
    
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(query_result, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Query log saved to: {log_path}")
    return log_path


def main():
    args = parse_args()
    
    query_result = query_manuals(
        question=args.question,
        collection_name=args.collection,
        top_k=args.top_k,
        prefetch_limit=args.prefetch_limit,
        device=args.device,
        manual_filter=args.manual_filter,
    )
    
    pretty_print_results(query_result)
    save_query_log(query_result)


if __name__ == "__main__":
    main()

