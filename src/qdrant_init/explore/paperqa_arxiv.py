#!/usr/bin/env python3
"""
Paperqa Arxiv Query Script

Query arxiv papers (PDFs) using paperqa with simple logging and progress tracking.

Usage Examples:

1. Query all PDFs in test directory with default ollama model:
    python paperqa_arxiv.py \
        --question "What is the main contribution of this paper?"

2. Query with custom model (e.g., Claude):
    python paperqa_arxiv.py \
        --question "What are the key findings?" \
        --model "claude-3-5-sonnet-20240620" \
        --max-sources 5

3. Query with custom ollama settings:
    python paperqa_arxiv.py \
        --question "What methodology is used?" \
        --model "ollama/llama3.2" \
        --ollama-api-base "http://localhost:11434" \
        --embedding-model "ollama/mxbai-embed-large"

Output Structure:
    logs/paperqa_arxiv/{timestamp}/
        query_metadata.json          # Overall query information
        results_summary.csv           # Summary of all results
        {pdf_name}_response.json      # Individual PDF response(s)
"""

import argparse
import asyncio
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from paperqa import Docs, Settings
from paperqa.settings import AgentSettings
from tqdm.asyncio import tqdm as async_tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Query arxiv papers (PDFs) using paperqa",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--question",
        required=True,
        help="The question to ask the paper(s)",
    )
    parser.add_argument(
        "--model",
        default="ollama/llama3.2",
        help="LLM model for paperqa (default: ollama/llama3.2)",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/paperqa_arxiv",
        help="Directory for logs (default: logs/paperqa_arxiv)",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=3,
        help="Maximum number of sources for answer (default: 3)",
    )
    parser.add_argument(
        "--pdf-dir",
        default="documents/gold_standard_pdfs/test",
        help="Path to PDFs directory (default: documents/gold_standard_pdfs/test)",
    )
    parser.add_argument(
        "--ollama-api-base",
        default="http://localhost:11434",
        help="Ollama API base URL (default: http://localhost:11434)",
    )
    parser.add_argument(
        "--embedding-model",
        default="ollama/mxbai-embed-large",
        help="Embedding model to use (default: ollama/mxbai-embed-large)",
    )
    return parser.parse_args()


def find_pdfs(pdf_dir: Path) -> list[Path]:
    """Find all PDF files in the specified directory."""
    if not pdf_dir.exists():
        raise FileNotFoundError(f"PDF directory not found: {pdf_dir}")
    
    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    
    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found in: {pdf_dir}")
    
    return pdf_files


async def query_single_pdf(
    pdf_path: Path,
    question: str,
    model: str,
    max_sources: int,
    ollama_api_base: str = "http://localhost:11434",
    embedding_model: str = "ollama/mxbai-embed-large",
) -> dict[str, Any]:
    """
    Query a single PDF using paperqa.
    
    Returns a dictionary with query results and metadata.
    """
    start_time = time.time()
    timestamp = datetime.now().isoformat()
    
    result = {
        "pdf_name": pdf_path.name,
        "pdf_path": str(pdf_path),
        "question": question,
        "model": model,
        "timestamp": timestamp,
        "status": None,
        "answer": None,
        "response_time_seconds": None,
        "contexts": [],
        "error": None,
    }
    
    try:
        # Create Docs object and add the PDF
        docs = Docs()
        await docs.aadd(str(pdf_path))
        
        # Set up paperqa settings
        # Use ollama configuration only for ollama models
        if model.startswith("ollama/"):
            # Set up local LLM configuration for ollama
            # CRITICAL: Must include "api_type": "ollama" to prevent OpenAI defaults
            local_llm_config = {
                "model_list": [
                    {
                        "model_name": model,
                        "litellm_params": {
                            "model": model,
                            "api_base": ollama_api_base,
                            "api_type": "ollama",  # Critical for ollama recognition
                        },
                    }
                ]
            }
            
            # Separate embedding config with api_type
            embedding_llm_config = {
                "model_list": [
                    {
                        "model_name": embedding_model,
                        "litellm_params": {
                            "model": embedding_model,
                            "api_base": ollama_api_base,
                            "api_type": "ollama",  # Critical for ollama recognition
                        },
                    }
                ]
            }
            
            # Create Settings object with AgentSettings configured for ollama
            settings = Settings(
                llm=model,
                llm_config=local_llm_config,
                summary_llm=model,
                summary_llm_config=local_llm_config,
                embedding=embedding_model,
                embedding_config=embedding_llm_config,
                agent=AgentSettings(
                    agent_llm=model,
                    agent_llm_config=local_llm_config,
                ),
            )
            
        else:
            # For cloud models (e.g., Claude), use default configuration
            settings = Settings(llm=model)
        
        settings.answer.answer_max_sources = max_sources
        
        # Query the document
        response = await docs.aquery(question, settings=settings)
        
        # Extract results
        result["status"] = response.status.value if hasattr(response, "status") else "success"
        result["answer"] = response.session.answer if hasattr(response, "session") else str(response)
        
        # Extract contexts
        if hasattr(response, "session") and hasattr(response.session, "contexts"):
            result["contexts"] = [
                {
                    "source": ctx.text.name if hasattr(ctx, "text") else "unknown",
                    "content": ctx.context if hasattr(ctx, "context") else "",
                    "score": ctx.score if hasattr(ctx, "score") else None,
                }
                for ctx in response.session.contexts
            ]
        
        # Store settings
        result["settings"] = {
            "llm": model,
            "answer_max_sources": max_sources,
        }
        
        # Add ollama-specific settings if using ollama
        if model.startswith("ollama/"):
            result["settings"]["embedding"] = embedding_model
            result["settings"]["ollama_api_base"] = ollama_api_base
    
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    finally:
        result["response_time_seconds"] = time.time() - start_time
    
    return result


def save_results(
    results: list[dict[str, Any]],
    output_dir: Path,
    run_timestamp: str,
    question: str,
    model: str,
    max_sources: int,
) -> None:
    """Save all results to JSON and CSV files."""
    # Create output directory
    run_dir = output_dir / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save query metadata
    metadata = {
        "run_timestamp": run_timestamp,
        "question": question,
        "model": model,
        "max_sources": max_sources,
        "pdf_count": len(results),
        "pdfs": [r["pdf_name"] for r in results],
    }
    
    metadata_file = run_dir / "query_metadata.json"
    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Save individual PDF responses
    for result in results:
        pdf_name = Path(result["pdf_name"]).stem
        response_file = run_dir / f"{pdf_name}_response.json"
        with response_file.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    
    # Save results summary CSV
    df_data = []
    for result in results:
        df_data.append({
            "pdf_name": result["pdf_name"],
            "question": result["question"],
            "answer": result["answer"],
            "status": result["status"],
            "response_time_seconds": result["response_time_seconds"],
            "timestamp": result["timestamp"],
            "error": result.get("error"),
        })
    
    df = pd.DataFrame(df_data)
    csv_file = run_dir / "results_summary.csv"
    df.to_csv(csv_file, index=False)


async def async_main(args):
    """Async main function to orchestrate the query process."""
    # Convert paths to Path objects
    output_dir = Path(args.output_dir)
    pdf_dir = Path(args.pdf_dir)
    
    # Create run timestamp
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("Paperqa Arxiv Query Script")
    print("=" * 50)
    print(f"Question: {args.question}")
    print(f"Model: {args.model}")
    print(f"Max sources: {args.max_sources}")
    print(f"PDF directory: {pdf_dir}")
    print(f"Output directory: {output_dir / run_timestamp}")
    print("=" * 50)
    
    # Find all PDFs
    try:
        pdf_files = find_pdfs(pdf_dir)
        print(f"\nFound {len(pdf_files)} PDF(s)")
    except Exception as e:
        print(f"Error finding PDFs: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Create async tasks for all PDFs
    tasks = []
    for pdf_path in pdf_files:
        task = query_single_pdf(
            pdf_path=pdf_path,
            question=args.question,
            model=args.model,
            max_sources=args.max_sources,
            ollama_api_base=args.ollama_api_base,
            embedding_model=args.embedding_model,
        )
        tasks.append(task)
    
    # Execute tasks with progress bar
    results = []
    print(f"\nQuerying {len(pdf_files)} PDFs...")
    for coro in async_tqdm.as_completed(tasks, total=len(tasks), desc="Progress"):
        result = await coro
        results.append(result)
    
    # Save results
    try:
        save_results(
            results=results,
            output_dir=output_dir,
            run_timestamp=run_timestamp,
            question=args.question,
            model=args.model,
            max_sources=args.max_sources,
        )
    except Exception as e:
        print(f"Error saving results: {e}", file=sys.stderr)
    
    # Display summary
    print(f"\n{'=' * 50}")
    print("Query complete!")
    print("=" * 50)
    
    successful = [r for r in results if r["status"] != "error"]
    failed = [r for r in results if r["status"] == "error"]
    
    print("\nSummary:")
    print(f"  Total PDFs: {len(results)}")
    print(f"  Successful: {len(successful)}")
    print(f"  Failed: {len(failed)}")
    
    if results:
        avg_time = sum(r["response_time_seconds"] for r in results) / len(results)
        print(f"  Average response time: {avg_time:.2f}s")
    
    print(f"\nResults saved to: {output_dir / run_timestamp}")
    print("  - query_metadata.json")
    print("  - results_summary.csv")
    print(f"  - {len(results)} individual response JSON files")
    
    # Print individual results
    print(f"\n{'=' * 50}")
    print("Individual Results:")
    print("=" * 50)
    
    for result in results:
        print(f"\nPDF: {result['pdf_name']}")
        print(f"Status: {result['status']}")
        print(f"Response time: {result['response_time_seconds']:.2f}s")
        
        if result.get("error"):
            print(f"Error: {result['error']}")
        else:
            print(f"Answer: {result['answer'][:200]}..." if len(result['answer']) > 200 else f"Answer: {result['answer']}")
            
            if result.get("contexts"):
                print(f"Sources: {len(result['contexts'])}")
    
    if failed:
        print(f"\n{'=' * 50}")
        print("Failed PDFs:")
        print("=" * 50)
        for result in failed:
            print(f"  - {result['pdf_name']}: {result['error']}")
    
    print(f"\n{'=' * 50}")
    print("Done!")


def main():
    """Entry point for the script."""
    args = parse_args()
    
    # Run async main function
    try:
        asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

