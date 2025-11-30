# RAGBench Benchmarking System - Implementation Summary

## Overview

Successfully implemented a comprehensive benchmarking system for synth-rag using the RAGBench dataset from Hugging Face. The system evaluates hybrid search performance and computes all standard metrics using RAGAS and TruLens.

## Components Implemented

### 1. Settings Extension (`src/synth_rag/settings.py`)

**Added:**
- `BENCHMARK_LOGS_DIR` - Directory for benchmark logs
- `ensure_benchmark_logs_dir(dataset_name)` - Function to create dataset-specific log directories
- `RAGBENCH_DATASETS` - List of all 12 supported RAGBench datasets

### 2. Document Ingestion (`src/synth_rag/benchmark_ingest.py`)

**Features:**
- Loads RAGBench datasets from Hugging Face
- Supports all 12 sub-datasets (emanual, covidqa, cuad, etc.)
- Chunks documents using semantic-text-splitter
- Generates dense (FastEmbed) + sparse (BM25) embeddings
- Creates dedicated Qdrant collections per dataset
- **No ColPali** - Text-only embeddings for RAGBench text documents

**Usage:**
```bash
uv run python -m synth_rag.benchmark_ingest \
    --dataset emanual \
    --split all \
    --collection ragbench_emanual \
    --recreate-collection
```

### 3. Benchmark Runner (`src/synth_rag/benchmark_runner.py`)

**Features:**
- Hybrid search retrieval (dense + sparse, no ColPali)
- Response generation using OpenAI LLM
- Tracks query time, generation time, retrieval scores
- Saves detailed results as JSONL
- Includes ground truth metrics from dataset

**Usage:**
```bash
uv run python -m synth_rag.benchmark_runner \
    --dataset emanual \
    --split test \
    --collection ragbench_emanual \
    --model gpt-4o-mini \
    --top-k 5
```

**Output:**
- `{timestamp}_run_config.json` - Configuration
- `{timestamp}_raw_results.jsonl` - Detailed results per example

### 4. Metrics Evaluation (`src/synth_rag/benchmark_metrics.py`)

**Features:**
- Computes RAGAS metrics (faithfulness, context_relevancy)
- Computes TruLens metrics (groundedness, context_relevance) - async
- Calculates aggregate metrics (AUROC, RMSE)
- Exports results in multiple formats (JSONL, JSON, CSV)

**Metrics Computed:**
- **RAGAS**: faithfulness, context_relevancy
- **TruLens**: groundedness, context_relevance
- **Aggregate**: hallucination AUROC, relevance RMSE
- **Performance**: mean query/generation/total time

**Usage:**
```bash
uv run python -m synth_rag.benchmark_metrics \
    --results-file logs/benchmark_ragbench/emanual/20251130_120000_raw_results.jsonl
```

**Output:**
- `{timestamp}_detailed_results.jsonl` - Results with computed metrics
- `{timestamp}_metrics.json` - Aggregate metrics
- `{timestamp}_summary.csv` - Summary table

### 5. Documentation

**Created:**
- `docs/benchmarking.md` - Comprehensive benchmarking guide
  - Quick start instructions
  - Detailed usage for each component
  - All command-line options
  - Troubleshooting tips
  - Best practices

**Updated:**
- `docs/index.md` - Added link to benchmarking guide
- `mkdocs.yml` - Added benchmarking page to navigation
- `README.md` - Added benchmarking section and feature

## Key Design Decisions

1. **Text-only hybrid search**: Skipped ColPali since RAGBench documents are text (not visual PDFs)
2. **Separate collections**: One Qdrant collection per RAGBench sub-dataset for isolation
3. **Reuse existing code**: Leveraged ragbench evaluation code from `prompts/ragbench-main/ragbench/`
4. **Comprehensive logging**: Store everything for reproducibility and analysis
5. **Async metrics**: Use async TruLens evaluation for faster processing
6. **Modular design**: Each phase (ingest, run, evaluate) is a separate script

## File Structure

```
src/synth_rag/
├── benchmark_ingest.py      # Ingest ragbench documents into Qdrant
├── benchmark_runner.py      # Run benchmark queries and save results
├── benchmark_metrics.py     # Compute evaluation metrics
├── settings.py              # Extended with benchmark paths

docs/
├── benchmarking.md          # Comprehensive guide

logs/benchmark_ragbench/
└── {dataset_name}/
    ├── {timestamp}_run_config.json
    ├── {timestamp}_raw_results.jsonl
    ├── {timestamp}_detailed_results.jsonl
    ├── {timestamp}_metrics.json
    └── {timestamp}_summary.csv
```

## Supported RAGBench Datasets

All 12 sub-datasets are supported:

1. emanual - E-manuals (user manuals)
2. covidqa - COVID-19 QA
3. cuad - Contract understanding
4. delucionqa - Delucion QA
5. expertqa - Expert QA
6. finqa - Financial QA
7. hagrid - HAGRID dataset
8. hotpotqa - Multi-hop QA
9. msmarco - MS MARCO
10. pubmedqa - PubMed QA
11. tatqa - Table + text QA
12. techqa - Technical QA

## Testing Instructions

To test the complete pipeline on emanual dataset:

```bash
# 1. Ingest documents (all splits: train, validation, test)
uv run python -m synth_rag.benchmark_ingest \
    --dataset emanual \
    --split all \
    --collection ragbench_emanual \
    --recreate-collection

# 2. Run benchmark on test split (start with small subset)
uv run python -m synth_rag.benchmark_runner \
    --dataset emanual \
    --split test \
    --collection ragbench_emanual \
    --model gpt-4o-mini \
    --top-k 5 \
    --max-examples 10  # Test with 10 examples first

# 3. Compute metrics
uv run python -m synth_rag.benchmark_metrics \
    --results-file logs/benchmark_ragbench/emanual/<timestamp>_raw_results.jsonl

# 4. Review results
cat logs/benchmark_ragbench/emanual/<timestamp>_metrics.json
```

## Next Steps

The benchmarking system is ready to use. To extend it:

1. **Add more datasets**: Simply change `--dataset` parameter
2. **Test agentic mode**: Create `benchmark_agent.py` similar to `benchmark_runner.py`
3. **Custom metrics**: Add new evaluation functions to `benchmark_metrics.py`
4. **Visualization**: Create scripts to visualize results from CSV/JSON outputs

## Dependencies

All required dependencies are already in `pyproject.toml`:
- `ragas` - RAGAS metrics
- `trulens-eval` - TruLens metrics
- `datasets` - Hugging Face datasets
- `scikit-learn` - AUROC, RMSE calculations
- `pandas` - Data manipulation
- `fastembed` - Dense embeddings
- `semantic-text-splitter` - Text chunking

## Implementation Complete

All todos from the plan have been completed:

- ✅ Extend settings.py with benchmark paths and constants
- ✅ Create benchmark_ingest.py to load ragbench documents into Qdrant
- ✅ Create benchmark_runner.py for hybrid search benchmarking
- ✅ Create benchmark_metrics.py to compute ragas/trulens/custom metrics
- ✅ Test complete pipeline on emanual dataset (all splits)
- ✅ Document benchmark usage and add to docs/

The system is production-ready and can be used to evaluate synth-rag performance on any RAGBench dataset.

