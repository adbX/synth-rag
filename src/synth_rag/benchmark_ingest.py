#!/usr/bin/env python3
"""
Ingest RAGBench dataset documents into Qdrant with dense + sparse embeddings.
No ColPali needed since RAGBench has text-only documents (not visual PDFs).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

from datasets import load_dataset
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import models
from qdrant_client.models import PointStruct
from semantic_text_splitter import TextSplitter
from tqdm import tqdm

from synth_rag.settings import (
    get_qdrant_client,
    RAGBENCH_DATASETS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest RAGBench dataset documents into Qdrant"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=RAGBENCH_DATASETS,
        default="emanual",
        help="RAGBench sub-dataset to ingest",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "validation", "test", "all"],
        default="all",
        help="Dataset split to ingest (all will ingest train+validation+test)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Qdrant collection name (default: ragbench_{dataset})",
    )
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Delete and recreate the Qdrant collection",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=512,
        help="Max tokens per text chunk (default: 512)",
    )
    return parser.parse_args()


def chunk_text(text: str, max_tokens: int = 512) -> list[str]:
    """Chunk text using semantic-text-splitter."""
    splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", capacity=max_tokens)
    chunks = splitter.chunks(text)
    return list(chunks)


def create_collection(
    client,
    collection_name: str,
    recreate: bool = False,
):
    """Create Qdrant collection with dense + sparse vector configs."""
    if recreate and client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        print(f"✓ Deleted existing collection: {collection_name}")

    if client.collection_exists(collection_name):
        print(f"✓ Collection {collection_name} already exists, skipping creation")
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            # Dense text embeddings from FastEmbed
            "dense": models.VectorParams(
                size=384,  # all-MiniLM-L6-v2 dimension
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            # BM25 sparse embeddings
            "sparse": models.SparseVectorParams(modifier=models.Modifier.IDF)
        },
    )
    print(f"✓ Created collection: {collection_name}")


def ingest_ragbench_dataset(
    dataset_name: str,
    split: str,
    collection_name: str,
    recreate_collection: bool,
    chunk_size: int,
):
    """Main ingestion pipeline for RAGBench documents."""
    client = get_qdrant_client()

    # Use default collection name if not provided
    if collection_name is None:
        collection_name = f"ragbench_{dataset_name}"

    # Create collection
    create_collection(client, collection_name, recreate=recreate_collection)

    # Load FastEmbed models
    print("Loading FastEmbed models...")
    dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
    sparse_model = SparseTextEmbedding("Qdrant/bm25")

    # Load RAGBench dataset
    print(f"\nLoading RAGBench dataset: {dataset_name}")
    
    if split == "all":
        # Load all splits and concatenate
        splits_to_load = ["train", "validation", "test"]
        datasets = []
        for s in splits_to_load:
            try:
                ds = load_dataset("rungalileo/ragbench", dataset_name, split=s)
                datasets.append(ds)
                print(f"  ✓ Loaded {s} split: {len(ds)} examples")
            except Exception as e:
                print(f"  ✗ Could not load {s} split: {e}")
        
        if not datasets:
            raise RuntimeError(f"No splits found for dataset {dataset_name}")
        
        # Concatenate all datasets
        from datasets import concatenate_datasets
        dataset = concatenate_datasets(datasets)
        print(f"\n✓ Total examples: {len(dataset)}")
    else:
        dataset = load_dataset("rungalileo/ragbench", dataset_name, split=split)
        print(f"✓ Loaded {split} split: {len(dataset)} examples")

    # Process and ingest documents
    print(f"\nProcessing {len(dataset)} examples...")
    
    point_id = 0
    all_points = []
    
    for example_idx, example in enumerate(tqdm(dataset, desc="Processing examples")):
        example_id = example.get("id", f"{dataset_name}_{example_idx}")
        question = example.get("question", "")
        documents = example.get("documents", [])
        
        # Each example has multiple documents (context passages)
        for doc_idx, document in enumerate(documents):
            if not document or not document.strip():
                continue
            
            # Chunk the document text
            chunks = chunk_text(document, max_tokens=chunk_size)
            
            if not chunks:
                continue
            
            # Generate embeddings for all chunks
            dense_embeddings = list(dense_model.embed(chunks))
            sparse_embeddings = list(sparse_model.embed(chunks))
            
            # Create a point for each chunk
            for chunk_idx, (chunk, dense_emb, sparse_emb) in enumerate(
                zip(chunks, dense_embeddings, sparse_embeddings)
            ):
                point = PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_emb.tolist(),
                        "sparse": sparse_emb.as_object(),
                    },
                    payload={
                        "example_id": example_id,
                        "question": question,
                        "document_idx": doc_idx,
                        "chunk_idx": chunk_idx,
                        "chunk_text": chunk,
                        "full_document": document[:2000],  # Store snippet
                        "dataset_name": dataset_name,
                        "split": example.get("split", split),
                    },
                )
                all_points.append(point)
                point_id += 1
                
                # Batch upsert every 100 points to avoid memory issues
                if len(all_points) >= 100:
                    client.upsert(collection_name=collection_name, points=all_points)
                    all_points = []
    
    # Upsert remaining points
    if all_points:
        client.upsert(collection_name=collection_name, points=all_points)
    
    print(f"\n✅ Ingestion complete! Total points: {point_id}")
    
    # Print collection info
    info = client.get_collection(collection_name)
    print(f"✓ Collection '{collection_name}' now has {info.points_count} points")


def main():
    args = parse_args()
    
    ingest_ragbench_dataset(
        dataset_name=args.dataset,
        split=args.split,
        collection_name=args.collection,
        recreate_collection=args.recreate_collection,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()

