#!/usr/bin/env python3
"""
Ingest PDF manuals into Qdrant with ColPali multivectors + FastEmbed dense/sparse embeddings.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Literal

import torch
import pypdfium2 as pdfium
import pymupdf
from PIL import Image
from tqdm import tqdm
from colpali_engine.models import ColPali, ColPaliProcessor
from fastembed import TextEmbedding, SparseTextEmbedding
from semantic_text_splitter import TextSplitter
from qdrant_client import models
from qdrant_client.models import PointStruct

from synth_rag.settings import (
    get_qdrant_client,
    get_manual_input_dir,
    ensure_tmp_dirs,
)

from qdrant_client.http.exceptions import UnexpectedResponse


def upsert_with_retry(
    client,
    collection_name: str,
    points: list,
    max_retries: int = 5,
    base_delay: float = 1.0,
):
    """Upsert points with exponential backoff retry on transient errors."""
    for attempt in range(max_retries):
        try:
            client.upsert(collection_name=collection_name, points=points)
            return
        except UnexpectedResponse as e:
            # Retry on 503 Service Unavailable or other transient errors
            if e.status_code in (503, 429, 502, 504) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"    Qdrant returned {e.status_code}, retrying in {delay:.1f}s...")
                time.sleep(delay)
            else:
                raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest PDF manuals with ColPali + FastEmbed")
    parser.add_argument(
        "--subset",
        type=str,
        choices=["test", "full"],
        default="test",
        help="Which subset of manuals to ingest (test or full)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="midi_manuals",
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--clear-tmp",
        action="store_true",
        help="Clear tmp directories before ingestion",
    )
    parser.add_argument(
        "--recreate-collection",
        action="store_true",
        help="Delete and recreate the Qdrant collection",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device for ColPali model (mps, cuda:0, cpu)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for ColPali embedding",
    )
    return parser.parse_args()


def render_pdf_pages(pdf_path: Path, output_dir: Path) -> list[Path]:
    """Render PDF pages to RGB images using pypdfium2."""
    pdf = pdfium.PdfDocument(pdf_path)
    page_paths = []
    
    manual_name = pdf_path.stem
    manual_dir = output_dir / manual_name
    manual_dir.mkdir(parents=True, exist_ok=True)
    
    for page_idx in range(len(pdf)):
        page = pdf[page_idx]
        # Render at 2x scale for better quality
        bitmap = page.render(scale=2.0)
        pil_image = bitmap.to_pil()
        
        page_path = manual_dir / f"page_{page_idx:04d}.png"
        pil_image.save(page_path, format="PNG")
        page_paths.append(page_path)
    
    return page_paths


def extract_page_text(pdf_path: Path) -> list[str]:
    """Extract text from each page using pymupdf."""
    doc = pymupdf.open(pdf_path)
    page_texts = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        page_texts.append(text)
    
    return page_texts


def save_text_manifest(manual_name: str, page_texts: list[str], output_dir: Path) -> Path:
    """Save page text as JSON manifest."""
    manifest_path = output_dir / f"{manual_name}.json"
    manifest = {
        "manual_name": manual_name,
        "pages": [
            {"page_num": i, "text": text}
            for i, text in enumerate(page_texts)
        ]
    }
    
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    return manifest_path


def mean_pool_colpali_embeddings(
    embeddings: torch.Tensor,
    input_ids: torch.Tensor,
    processor: ColPaliProcessor,
    model: ColPali,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Mean pool ColPali embeddings by rows and columns.
    Returns: (original, pooled_rows, pooled_cols)
    """
    # embeddings shape: (batch_size, num_vectors, dim)
    # For ColPali: (batch_size, 1030, 128)
    
    batch_originals = []
    batch_rows = []
    batch_cols = []
    
    for batch_idx in range(embeddings.shape[0]):
        embedding = embeddings[batch_idx]  # (1030, 128)
        
        # Identify image tokens
        mask = input_ids[batch_idx] == processor.image_token_id
        
        # ColPali always uses 32x32 patches
        x_patches, y_patches = 32, 32
        
        # Extract image patch embeddings
        image_patch_embeddings = embedding[mask].view(x_patches, y_patches, model.dim)
        
        # Mean pool by rows and columns
        pooled_rows = image_patch_embeddings.mean(dim=1)  # (x_patches, 128)
        pooled_cols = image_patch_embeddings.mean(dim=0)  # (y_patches, 128)
        
        # Concatenate special tokens (postfix for ColPali)
        special_tokens = embedding[~mask]
        pooled_rows = torch.cat([pooled_rows, special_tokens])
        pooled_cols = torch.cat([pooled_cols, special_tokens])
        
        batch_originals.append(embedding)
        batch_rows.append(pooled_rows)
        batch_cols.append(pooled_cols)
    
    return batch_originals, batch_rows, batch_cols


def create_collection(client, collection_name: str, recreate: bool = False):
    """Create Qdrant collection with multivector + dense + sparse configs."""
    if recreate and client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        print(f"Deleted existing collection: {collection_name}")
    
    if client.collection_exists(collection_name):
        print(f"Collection {collection_name} already exists, skipping creation")
        return
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            # Original ColPali multivectors (no HNSW for reranking only)
            "colpali_original": models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
                hnsw_config=models.HnswConfigDiff(m=0),  # Disable HNSW
            ),
            # Mean-pooled rows for first-stage retrieval
            "colpali_rows": models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
            # Mean-pooled columns for first-stage retrieval
            "colpali_cols": models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
            # Dense text embeddings from FastEmbed
            "dense": models.VectorParams(
                size=384,  # all-MiniLM-L6-v2 dimension
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            # BM25 sparse embeddings
            "sparse": models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        },
    )
    print(f"Created collection: {collection_name}")


def chunk_text(text: str, max_tokens: int = 256) -> list[str]:
    """Chunk text using semantic-text-splitter."""
    splitter = TextSplitter.from_tiktoken_model("gpt-3.5-turbo", capacity=max_tokens)
    chunks = splitter.chunks(text)
    return list(chunks)


def ingest_manuals(
    subset: Literal["test", "full"],
    collection_name: str,
    device: str,
    batch_size: int,
    clear_tmp: bool,
    recreate_collection: bool,
):
    """Main ingestion pipeline."""
    # Setup
    client = get_qdrant_client()
    input_dir = get_manual_input_dir(subset)
    pages_dir, text_dir = ensure_tmp_dirs(clear=clear_tmp)
    
    # Create collection
    create_collection(client, collection_name, recreate=recreate_collection)
    
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
    
    # Get all PDFs
    pdf_files = sorted(input_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in {input_dir}")
    
    point_id = 0
    
    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        manual_name = pdf_path.stem
        
        # Render pages
        print(f"\n  Rendering pages for {manual_name}...")
        page_image_paths = render_pdf_pages(pdf_path, pages_dir)
        
        # Extract text
        print("  Extracting text...")
        page_texts = extract_page_text(pdf_path)
        
        # Save text manifest
        save_text_manifest(manual_name, page_texts, text_dir)
        
        # Process pages in batches for ColPali
        print("  Generating ColPali embeddings...")
        all_page_embeddings = []
        
        for i in range(0, len(page_image_paths), batch_size):
            batch_paths = page_image_paths[i:i + batch_size]
            batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
            
            # Generate ColPali embeddings
            processed_images = colpali_processor.process_images(batch_images).to(device)
            with torch.no_grad():
                image_embeddings = colpali_model(**processed_images)
            
            # Mean pool
            originals, rows, cols = mean_pool_colpali_embeddings(
                image_embeddings,
                processed_images.input_ids,
                colpali_processor,
                colpali_model,
            )
            
            for j, (orig, row, col) in enumerate(zip(originals, rows, cols)):
                page_idx = i + j
                all_page_embeddings.append({
                    "page_num": page_idx,
                    "original": orig.cpu().float().numpy().tolist(),
                    "rows": row.cpu().float().numpy().tolist(),
                    "cols": col.cpu().float().numpy().tolist(),
                    "text": page_texts[page_idx],
                    "image_path": str(batch_paths[j]),
                })
        
        # Now process text chunks for dense/sparse embeddings
        print("  Processing text chunks...")
        points = []
        
        for page_data in all_page_embeddings:
            page_num = page_data["page_num"]
            page_text = page_data["text"]
            
            # Chunk the page text
            if page_text.strip():
                chunks = chunk_text(page_text)
            else:
                chunks = [""]  # Empty page
            
            # Generate dense and sparse embeddings for chunks
            dense_embeddings = list(dense_model.embed(chunks))
            sparse_embeddings = list(sparse_model.embed(chunks))
            
            # Create points (one per page, with first chunk's text embedding)
            # We align by using the first chunk for simplicity
            point = PointStruct(
                id=point_id,
                vector={
                    "colpali_original": page_data["original"],
                    "colpali_rows": page_data["rows"],
                    "colpali_cols": page_data["cols"],
                    "dense": dense_embeddings[0].tolist() if dense_embeddings else [0.0] * 384,
                    "sparse": sparse_embeddings[0].as_object() if sparse_embeddings else models.SparseVector(indices=[], values=[]),
                },
                payload={
                    "manual_name": manual_name,
                    "page_num": page_num,
                    "text": page_text[:1000],  # Store snippet
                    "full_text": page_text,
                    "image_path": page_data["image_path"],
                    "num_chunks": len(chunks),
                },
            )
            points.append(point)
            point_id += 1
        
        # Upsert in smaller batches to avoid payload size limits
        print(f"  Upserting {len(points)} points in batches...")
        upsert_batch_size = 10  # Keep payload size manageable
        for i in range(0, len(points), upsert_batch_size):
            batch = points[i:i + upsert_batch_size]
            upsert_with_retry(client, collection_name, batch)
            print(f"    Upserted {min(i + upsert_batch_size, len(points))}/{len(points)} points")
    
    print(f"\n✅ Ingestion complete! Total points: {point_id}")
    
    # Print collection info
    info = client.get_collection(collection_name)
    print(f"Collection '{collection_name}' now has {info.points_count} points")


def main():
    args = parse_args()
    ingest_manuals(
        subset=args.subset,
        collection_name=args.collection,
        device=args.device,
        batch_size=args.batch_size,
        clear_tmp=args.clear_tmp,
        recreate_collection=args.recreate_collection,
    )


if __name__ == "__main__":
    main()

