#!/usr/bin/env python3
"""
Paperqa Computo Query Script

Query Computo papers using paperqa with robust logging and progress tracking.

Usage Examples:

1. Query a single paper:
    python paperqa_computo.py \
        --question "What is the main contribution of this paper?" \
        --paper "published-202301-boulin-clayton" \
        --model "claude-3-5-sonnet-20240620"

2. Query all papers:
    python paperqa_computo.py \
        --question "What methodology does this paper use?" \
        --model "claude-3-5-sonnet-20240620"

3. Query with custom settings:
    python paperqa_computo.py \
        --question "What are the key findings?" \
        --model "claude-3-5-sonnet-20240620" \
        --max-sources 5 \
        --output-dir "my_results"

Output Structure:
    logs/paperqa/{timestamp}/
        query_metadata.json          # Overall query information
        results_summary.csv           # Summary of all results (batch mode only)
        {paper_name}_response.json    # Individual paper response(s)
        errors.log                    # Error log (if any errors occurred)
"""

import argparse
import asyncio
import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from paperqa import Docs, Settings
from tqdm.asyncio import tqdm as async_tqdm


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Query Computo papers using paperqa",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--question",
        required=True,
        help="The question to ask the paper(s)",
    )
    parser.add_argument(
        "--paper",
        help="Specific paper name from CSV to query (if omitted, queries all papers)",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="LLM model for paperqa (e.g., claude-3-5-sonnet-20240620)",
    )
    parser.add_argument(
        "--output-dir",
        default="logs/paperqa",
        help="Directory for logs (default: logs/paperqa)",
    )
    parser.add_argument(
        "--max-sources",
        type=int,
        default=3,
        help="Maximum number of sources for answer (default: 3)",
    )
    parser.add_argument(
        "--computo-csv",
        default="documents/computo/computo_repos.csv",
        help="Path to computo_repos.csv (default: documents/computo/computo_repos.csv)",
    )
    parser.add_argument(
        "--computo-docs-dir",
        default="documents/computo",
        help="Path to computo documents directory (default: documents/computo)",
    )
    return parser.parse_args()


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


def read_computo_papers(csv_path: Path) -> pd.DataFrame:
    """Read the computo_repos.csv file and return as DataFrame."""
    try:
        df = pd.read_csv(csv_path)
        return df
    except Exception as e:
        raise RuntimeError(f"Failed to read CSV file {csv_path}: {e}")


def find_qmd_files(paper_name: str, computo_docs_dir: Path) -> list[Path]:
    """
    Find and combine .qmd files for a given paper.
    
    Returns list of .qmd files, with main file first and supplementary files after.
    """
    paper_dir = computo_docs_dir / paper_name
    
    if not paper_dir.exists():
        raise FileNotFoundError(f"Paper directory not found: {paper_dir}")
    
    # Find main .qmd file (matches paper name)
    main_qmd = paper_dir / f"{paper_name}.qmd"
    
    if not main_qmd.exists():
        raise FileNotFoundError(f"Main .qmd file not found: {main_qmd}")
    
    qmd_files = [main_qmd]
    
    # Find supplementary .qmd files
    for qmd_file in sorted(paper_dir.glob("*.qmd")):
        if qmd_file != main_qmd:
            qmd_files.append(qmd_file)
    
    return qmd_files


def combine_qmd_files(qmd_files: list[Path]) -> str:
    """
    Combine multiple .qmd files into a single text string.
    
    Adds separator comments between files for clarity.
    """
    combined_text = []
    
    for qmd_file in qmd_files:
        try:
            content = qmd_file.read_text(encoding="utf-8")
            if len(qmd_files) > 1:
                combined_text.append(f"\n\n<!-- BEGIN: {qmd_file.name} -->\n\n")
            combined_text.append(content)
            if len(qmd_files) > 1:
                combined_text.append(f"\n\n<!-- END: {qmd_file.name} -->\n\n")
        except Exception as e:
            raise RuntimeError(f"Failed to read {qmd_file}: {e}")
    
    return "".join(combined_text)


async def query_single_paper(
    paper_name: str,
    question: str,
    model: str,
    max_sources: int,
    computo_docs_dir: Path,
) -> dict[str, Any]:
    """
    Query a single paper using paperqa.
    
    Returns a dictionary with query results and metadata.
    """
    start_time = time.time()
    timestamp = datetime.now().isoformat()
    
    result = {
        "paper_name": paper_name,
        "question": question,
        "model": model,
        "timestamp": timestamp,
        "status": None,
        "answer": None,
        "response_time_seconds": None,
        "contexts": [],
        "error": None,
        "qmd_files": [],
    }
    
    try:
        # Find and combine .qmd files
        qmd_files = find_qmd_files(paper_name, computo_docs_dir)
        result["qmd_files"] = [str(f) for f in qmd_files]
        
        combined_content = combine_qmd_files(qmd_files)
        
        # Create a temporary file with .md extension for paperqa
        # paperqa doesn't officially support .qmd, but we'll try .md
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as tmp_file:
            tmp_file.write(combined_content)
            tmp_path = Path(tmp_file.name)
        
        try:
            # Create Docs object and add the document
            docs = Docs()
            await docs.aadd(str(tmp_path))
            
            # Set up paperqa settings
            settings = Settings()
            settings.llm = model
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
            
            # Store full session data (convert to dict for JSON serialization)
            if hasattr(response, "session"):
                result["session_data"] = {
                    "question": response.session.question if hasattr(response.session, "question") else question,
                    "answer": response.session.answer if hasattr(response.session, "answer") else "",
                    "context_count": len(response.session.contexts) if hasattr(response.session, "contexts") else 0,
                }
            
            # Store settings
            result["settings"] = {
                "llm": model,
                "answer_max_sources": max_sources,
            }
            
        finally:
            # Clean up temporary file
            tmp_path.unlink(missing_ok=True)
    
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
    
    finally:
        result["response_time_seconds"] = time.time() - start_time
    
    return result


def save_query_log(
    result: dict[str, Any],
    output_dir: Path,
    run_timestamp: str,
) -> Path:
    """
    Save a single query result to a JSON file.
    
    Returns the path to the saved file.
    """
    # Create output directory structure: logs/paperqa/{timestamp}/
    run_dir = output_dir / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual paper response
    paper_name = result["paper_name"]
    response_file = run_dir / f"{paper_name}_response.json"
    
    with response_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    return response_file


def save_query_metadata(
    question: str,
    model: str,
    max_sources: int,
    output_dir: Path,
    run_timestamp: str,
    paper_names: list[str],
) -> Path:
    """
    Save overall query metadata to a JSON file.
    
    Returns the path to the saved file.
    """
    run_dir = output_dir / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    metadata = {
        "run_timestamp": run_timestamp,
        "question": question,
        "model": model,
        "max_sources": max_sources,
        "paper_count": len(paper_names),
        "papers": paper_names,
    }
    
    metadata_file = run_dir / "query_metadata.json"
    
    with metadata_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    return metadata_file


def save_error_log(
    error: str,
    output_dir: Path,
    run_timestamp: str,
    paper_name: Optional[str] = None,
) -> Path:
    """
    Save error information to a log file.
    
    Returns the path to the saved file.
    """
    run_dir = output_dir / run_timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    
    error_file = run_dir / "errors.log"
    
    timestamp = datetime.now().isoformat()
    error_entry = f"[{timestamp}]"
    if paper_name:
        error_entry += f" [{paper_name}]"
    error_entry += f" {error}\n"
    
    with error_file.open("a", encoding="utf-8") as f:
        f.write(error_entry)
    
    return error_file


async def query_all_papers(
    question: str,
    model: str,
    max_sources: int,
    computo_docs_dir: Path,
    output_dir: Path,
    run_timestamp: str,
    paper_names: list[str],
) -> pd.DataFrame:
    """
    Query all papers with progress tracking.
    
    Returns a DataFrame with results for all papers.
    """
    results = []
    
    # Create async tasks for all papers
    tasks = []
    for paper_name in paper_names:
        task = query_single_paper(
            paper_name=paper_name,
            question=question,
            model=model,
            max_sources=max_sources,
            computo_docs_dir=computo_docs_dir,
        )
        tasks.append(task)
    
    # Execute tasks with progress bar
    print(f"\nQuerying {len(paper_names)} papers...")
    for coro in async_tqdm.as_completed(tasks, total=len(tasks), desc="Progress"):
        result = await coro
        
        # Save individual result
        try:
            save_query_log(result, output_dir, run_timestamp)
        except Exception as e:
            save_error_log(
                f"Failed to save log: {e}",
                output_dir,
                run_timestamp,
                result.get("paper_name"),
            )
        
        # Log errors if query failed
        if result.get("error"):
            save_error_log(
                result["error"],
                output_dir,
                run_timestamp,
                result["paper_name"],
            )
        
        results.append(result)
    
    # Create DataFrame with key information
    df_data = []
    for result in results:
        df_data.append({
            "paper_name": result["paper_name"],
            "question": result["question"],
            "answer": result["answer"],
            "status": result["status"],
            "response_time_seconds": result["response_time_seconds"],
            "timestamp": result["timestamp"],
            "error": result.get("error"),
        })
    
    df = pd.DataFrame(df_data)
    
    # Save DataFrame as CSV
    run_dir = output_dir / run_timestamp
    csv_file = run_dir / "results_summary.csv"
    df.to_csv(csv_file, index=False)
    
    return df


async def async_main(args):
    """Async main function to orchestrate the query process."""
    # Convert paths to Path objects
    output_dir = Path(args.output_dir)
    csv_path = Path(args.computo_csv)
    computo_docs_dir = Path(args.computo_docs_dir)
    
    # Create run timestamp
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"Paperqa Computo Query Script")
    print(f"=" * 50)
    print(f"Question: {args.question}")
    print(f"Model: {args.model}")
    print(f"Max sources: {args.max_sources}")
    print(f"Output directory: {output_dir / run_timestamp}")
    print(f"=" * 50)
    
    # Read papers from CSV
    try:
        papers_df = read_computo_papers(csv_path)
        all_paper_names = papers_df["repo_name"].tolist()
    except Exception as e:
        print(f"Error reading papers CSV: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Determine which papers to query
    if args.paper:
        # Query single paper
        if args.paper not in all_paper_names:
            print(f"Error: Paper '{args.paper}' not found in CSV", file=sys.stderr)
            print(f"Available papers: {', '.join(all_paper_names)}")
            sys.exit(1)
        
        paper_names = [args.paper]
        print(f"\nQuerying single paper: {args.paper}")
    else:
        # Query all papers
        paper_names = all_paper_names
        print(f"\nQuerying all {len(paper_names)} papers")
    
    # Save query metadata
    try:
        save_query_metadata(
            question=args.question,
            model=args.model,
            max_sources=args.max_sources,
            output_dir=output_dir,
            run_timestamp=run_timestamp,
            paper_names=paper_names,
        )
    except Exception as e:
        print(f"Warning: Failed to save metadata: {e}", file=sys.stderr)
    
    # Execute queries
    if len(paper_names) == 1:
        # Single paper query
        paper_name = paper_names[0]
        print(f"\nQuerying paper: {paper_name}")
        
        result = await query_single_paper(
            paper_name=paper_name,
            question=args.question,
            model=args.model,
            max_sources=args.max_sources,
            computo_docs_dir=computo_docs_dir,
        )
        
        # Save result
        try:
            log_file = save_query_log(result, output_dir, run_timestamp)
            print(f"\nResults saved to: {log_file}")
        except Exception as e:
            print(f"Error saving log: {e}", file=sys.stderr)
        
        # Display result
        print(f"\n{'=' * 50}")
        print(f"Paper: {result['paper_name']}")
        print(f"Status: {result['status']}")
        print(f"Response time: {result['response_time_seconds']:.2f}s")
        print(f"{'=' * 50}")
        
        if result.get("error"):
            print(f"\nError: {result['error']}")
        else:
            print(f"\nAnswer:\n{result['answer']}")
            
            if result.get("contexts"):
                print(f"\nContexts ({len(result['contexts'])}):")
                for i, ctx in enumerate(result["contexts"], 1):
                    print(f"  {i}. Source: {ctx['source']}, Score: {ctx['score']}")
    
    else:
        # Batch query all papers
        df = await query_all_papers(
            question=args.question,
            model=args.model,
            max_sources=args.max_sources,
            computo_docs_dir=computo_docs_dir,
            output_dir=output_dir,
            run_timestamp=run_timestamp,
            paper_names=paper_names,
        )
        
        # Display summary
        print(f"\n{'=' * 50}")
        print(f"Query complete!")
        print(f"{'=' * 50}")
        print(f"\nSummary:")
        print(f"  Total papers: {len(df)}")
        print(f"  Successful: {len(df[df['status'] != 'error'])}")
        print(f"  Failed: {len(df[df['status'] == 'error'])}")
        print(f"  Average response time: {df['response_time_seconds'].mean():.2f}s")
        print(f"\nResults saved to: {output_dir / run_timestamp}")
        print(f"  - query_metadata.json")
        print(f"  - results_summary.csv")
        print(f"  - {len(paper_names)} individual response JSON files")
        
        if len(df[df['status'] == 'error']) > 0:
            print(f"\nFailed papers:")
            for _, row in df[df['status'] == 'error'].iterrows():
                print(f"  - {row['paper_name']}: {row['error']}")
    
    print(f"\n{'=' * 50}")
    print("Done!")


if __name__ == "__main__":
    main()

