<!-- cc2aa300-68dc-40be-b809-9001da347583 e4184172-1f55-4625-9f48-5113d7ea6068 -->
# RAGBench Benchmarking Implementation Plan

## Overview

Create a comprehensive benchmarking system to evaluate synth-rag's hybrid search performance against the ragbench emanual dataset, computing all standard metrics (ragas faithfulness/context_relevance, trulens groundedness/context_relevance, adherence, relevance, utilization scores).

## Architecture

The benchmarking system will be independent of the main synth-rag functionality with these key components:

1. **Document Ingestion** ([`src/synth_rag/benchmark_ingest.py`](src/synth_rag/benchmark_ingest.py))

   - Load ragbench dataset documents (emanual or other sub-datasets)
   - Convert text documents to pseudo-PDF format or ingest as text chunks
   - Create dedicated Qdrant collection per sub-dataset (e.g., `ragbench_emanual`)
   - Reuse existing FastEmbed dense/sparse embeddings (no ColPali needed since ragbench has text-only documents)

2. **Benchmark Runner** ([`src/synth_rag/benchmark_runner.py`](src/synth_rag/benchmark_runner.py))

   - Load ragbench dataset (train/val/test splits)
   - For each question: retrieve relevant contexts using hybrid search
   - Generate response using retrieved contexts + LLM
   - Track retrieval metrics (latency, top-k scores)
   - Save raw results to structured logs

3. **Metrics Evaluation** ([`src/synth_rag/benchmark_metrics.py`](src/synth_rag/benchmark_metrics.py))

   - Load benchmark results from logs
   - Compute ragas metrics (faithfulness, context_relevancy) using existing ragbench code
   - Compute trulens metrics (groundedness, context_relevance) using async evaluation
   - Calculate custom metrics (adherence_score, relevance_score, utilization_score, completeness_score)
   - Aggregate and compare against ragbench ground truth
   - Export results as JSON and CSV

4. **Settings & Configuration** (extend [`src/synth_rag/settings.py`](src/synth_rag/settings.py))

   - Add benchmark-specific paths and constants
   - Support configurable dataset selection (emanual, covidqa, etc.)
   - Logging directory structure for benchmarks

5. **Comprehensive Logging** 

   - Create `logs/benchmark_ragbench/` directory structure
   - Per-run logs with timestamps
   - Store: query, retrieved_contexts, generated_response, ground_truth, metrics, parameters

## Implementation Details

### Phase 1: Document Ingestion

- **Key Decision**: Since ragbench has text documents (not PDFs), we'll:
  - Extract documents from the dataset's `documents` field
  - Create text chunks using semantic-text-splitter (same as manuals)
  - Generate dense (FastEmbed) + sparse (BM25) embeddings
  - Skip ColPali embeddings (text-only, no visual component)
  - Store in collection `ragbench_{dataset_name}` (e.g., `ragbench_emanual`)

### Phase 2: Hybrid Search Retrieval

- Use existing hybrid search from [`manuals_query.py`](src/synth_rag/manuals_query.py)
- Adapt to work without ColPali (dense + sparse only)
- Track retrieval metrics: query_time, top_k_scores, retrieved_context_ids

### Phase 3: Response Generation

- Use OpenAI LLM (gpt-4o-mini or configurable) to generate responses
- Prompt template: "Answer the question based on the provided context"
- Return: generated_response + metadata

### Phase 4: Metrics Evaluation

- Leverage existing ragbench evaluation code in `prompts/ragbench-main/ragbench/`
- Import and use:
  - `ragas_annotate_dataset()` from `inference.py`
  - `trulens_annotate_dataset()` from `inference.py`
  - Custom scoring functions from `evaluation.py`
- Compute all metrics:
  - **Ragas**: faithfulness, context_relevancy
  - **Trulens**: groundedness, context_relevance (async)
  - **Custom**: adherence_score, relevance_score, utilization_score, completeness_score
  - **Retrieval**: AUROC for hallucination detection, RMSE for relevance/utilization

### Phase 5: Extensibility

- Make dataset name configurable (default: emanual)
- Support all ragbench sub-datasets with minimal changes
- CLI flags: `--dataset {emanual,covidqa,cuad,...}`, `--split {train,validation,test,all}`

## File Structure

```
src/synth_rag/
├── benchmark_ingest.py      # Ingest ragbench documents into Qdrant
├── benchmark_runner.py      # Run benchmark queries and save results
├── benchmark_metrics.py     # Compute evaluation metrics
├── settings.py              # Extended with benchmark paths

logs/benchmark_ragbench/
├── {dataset_name}/
│   ├── {timestamp}_run_config.json
│   ├── {timestamp}_raw_results.jsonl
│   ├── {timestamp}_metrics.json
│   └── {timestamp}_summary.csv
```

## Usage Examples

```bash
# 1. Ingest ragbench emanual documents
uv run python -m synth_rag.benchmark_ingest \
    --dataset emanual \
    --split all \
    --collection ragbench_emanual \
    --recreate-collection

# 2. Run benchmark on test split
uv run python -m synth_rag.benchmark_runner \
    --dataset emanual \
    --split test \
    --collection ragbench_emanual \
    --model gpt-4o-mini \
    --top-k 5

# 3. Compute metrics on benchmark results
uv run python -m synth_rag.benchmark_metrics \
    --results-file logs/benchmark_ragbench/emanual/20251130_120000_raw_results.jsonl \
    --compute-all-metrics
```

## Key Design Decisions

1. **Text-only hybrid search**: Skip ColPali since ragbench documents are text (not visual PDFs)
2. **Separate collections**: One Qdrant collection per ragbench sub-dataset for isolation
3. **Reuse existing code**: Leverage ragbench evaluation code and synth-rag retrieval
4. **Comprehensive logging**: Store everything for reproducibility and analysis
5. **Async metrics**: Use async trulens evaluation for faster processing
6. **Modular design**: Each phase (ingest, run, evaluate) is a separate script

### To-dos

- [ ] Extend settings.py with benchmark paths and constants
- [ ] Create benchmark_ingest.py to load ragbench documents into Qdrant
- [ ] Create benchmark_runner.py for hybrid search benchmarking
- [ ] Create benchmark_metrics.py to compute ragas/trulens/custom metrics
- [ ] Test complete pipeline on emanual dataset (all splits)
- [ ] Document benchmark usage and add to docs/