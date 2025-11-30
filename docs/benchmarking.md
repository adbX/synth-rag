# RAGBench Benchmarking Guide

## Overview

This guide describes how to benchmark the synth-rag system using the [RAGBench dataset](https://huggingface.co/datasets/rungalileo/ragbench) from Hugging Face. The benchmarking system evaluates hybrid search performance and computes comprehensive metrics including RAGAS faithfulness/context_relevancy and TruLens groundedness/context_relevance.

## Architecture

The benchmarking system consists of three independent modules:

1. **Document Ingestion** ([`benchmark_ingest.py`](../src/synth_rag/benchmark_ingest.py)) - Loads RAGBench documents into a dedicated Qdrant collection
2. **Benchmark Runner** ([`benchmark_runner.py`](../src/synth_rag/benchmark_runner.py)) - Runs hybrid search queries and generates responses
3. **Metrics Evaluation** ([`benchmark_metrics.py`](../src/synth_rag/benchmark_metrics.py)) - Computes evaluation metrics using RAGAS and TruLens

### Key Differences from Main System

- **Text-only embeddings**: Uses FastEmbed (dense) + BM25 (sparse) only, no ColPali
- **Separate collections**: Each RAGBench sub-dataset gets its own Qdrant collection
- **Comprehensive logging**: All queries, responses, and metrics are saved for analysis

---

## Quick Start

### Prerequisites

Ensure you have the required dependencies installed:

```bash
# Already included in pyproject.toml:
# - ragas
# - trulens_eval
# - datasets
```

Set up your environment variables in `.env`:

```bash
QDRANT_URL=<your-qdrant-url>
QDRANT_KEY=<your-qdrant-key>
OPENAI_API_KEY=<your-openai-key>
```

### Complete Workflow Example

```bash
# 1. Ingest RAGBench emanual dataset (all splits)
uv run python -m synth_rag.benchmark_ingest \
    --dataset emanual \
    --split all \
    --collection ragbench_emanual \
    --recreate-collection

# 2. Run benchmark on test split (recommended to start small)
uv run python -m synth_rag.benchmark_runner \
    --dataset emanual \
    --split test \
    --collection ragbench_emanual \
    --model gpt-4o-mini \
    --top-k 5 \
    --max-examples 10  # Optional: limit for testing

# 3. Compute metrics on the results
uv run python -m synth_rag.benchmark_metrics \
    --results-file logs/benchmark_ragbench/emanual/20251130_120000_raw_results.jsonl
```

---

## Step 1: Document Ingestion

### Command

```bash
uv run python -m synth_rag.benchmark_ingest \
    --dataset emanual \
    --split all \
    --collection ragbench_emanual \
    --recreate-collection
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dataset` | str | emanual | RAGBench sub-dataset (emanual, covidqa, cuad, etc.) |
| `--split` | str | all | Dataset split (train, validation, test, all) |
| `--collection` | str | ragbench_{dataset} | Qdrant collection name |
| `--recreate-collection` | flag | False | Delete and recreate collection |
| `--chunk-size` | int | 512 | Max tokens per text chunk |

### What It Does

1. Loads the specified RAGBench dataset from Hugging Face
2. Extracts documents from each example
3. Chunks documents using semantic-text-splitter
4. Generates dense (FastEmbed) and sparse (BM25) embeddings
5. Uploads to Qdrant with metadata (example_id, question, document_idx, etc.)

### Output

```
✓ Loaded train split: 1054 examples
✓ Loaded validation split: 132 examples
✓ Loaded test split: 132 examples

✓ Total examples: 1318
Processing examples: 100%|████████████| 1318/1318
✅ Ingestion complete! Total points: 15234
✓ Collection 'ragbench_emanual' now has 15234 points
```

---

## Step 2: Running the Benchmark

### Command

```bash
uv run python -m synth_rag.benchmark_runner \
    --dataset emanual \
    --split test \
    --collection ragbench_emanual \
    --model gpt-4o-mini \
    --top-k 5
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--dataset` | str | emanual | RAGBench sub-dataset |
| `--split` | str | test | Dataset split to benchmark |
| `--collection` | str | ragbench_{dataset} | Qdrant collection name |
| `--model` | str | gpt-4o-mini | OpenAI model for response generation |
| `--top-k` | int | 5 | Number of contexts to retrieve |
| `--prefetch-limit` | int | 50 | Prefetch limit for reranking |
| `--max-examples` | int | None | Limit number of examples (for testing) |

### What It Does

For each example in the dataset:

1. **Retrieve contexts**: Uses hybrid search (dense + sparse) to find top-k relevant chunks
2. **Generate response**: Uses LLM with retrieved contexts to answer the question
3. **Track metrics**: Records query time, generation time, retrieval scores
4. **Save results**: Writes results as JSONL for later evaluation

### Output

Results are saved to `logs/benchmark_ragbench/{dataset_name}/`:

- `{timestamp}_run_config.json` - Configuration and metadata
- `{timestamp}_raw_results.jsonl` - Detailed results (one JSON object per line)

Example result entry:

```json
{
  "example_id": "emanual_265",
  "question": "How do I select Motion Lighting?",
  "ground_truth_documents": ["..."],
  "ground_truth_response": "...",
  "retrieved_contexts": ["...", "...", "..."],
  "retrieval_scores": [0.892, 0.856, 0.824],
  "retrieval_metadata": [...],
  "query_time_seconds": 0.234,
  "generated_response": "To select Motion Lighting...",
  "generation_time_seconds": 1.456,
  "total_time_seconds": 1.690,
  "ground_truth_adherence_score": true,
  "ground_truth_relevance_score": 0.95,
  "ground_truth_utilization_score": 0.88,
  "ground_truth_completeness_score": 1.0
}
```

---

## Step 3: Computing Metrics

### Command

```bash
uv run python -m synth_rag.benchmark_metrics \
    --results-file logs/benchmark_ragbench/emanual/20251130_120000_raw_results.jsonl
```

### Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `--results-file` | Path | Required | Path to JSONL results file |
| `--skip-ragas` | flag | False | Skip RAGAS metrics computation |
| `--skip-trulens` | flag | False | Skip TruLens metrics computation |
| `--max-concurrent` | int | 10 | Max concurrent requests for TruLens |

### What It Does

1. **Load results**: Reads JSONL results file
2. **RAGAS evaluation**: Computes faithfulness and context_relevancy
3. **TruLens evaluation**: Computes groundedness and context_relevance (async)
4. **Aggregate metrics**: Calculates means, standard deviations, AUROC, RMSE
5. **Save outputs**: Exports detailed results, aggregate metrics, and summary CSV

### Metrics Computed

#### RAGAS Metrics
- **Faithfulness**: Measures how grounded the generated response is in the retrieved contexts
- **Context Relevancy**: Measures how relevant the retrieved contexts are to the question

#### TruLens Metrics
- **Groundedness**: Similar to faithfulness, checks if response is supported by context
- **Context Relevance**: Evaluates if retrieved contexts contain information to answer the question

#### Aggregate Metrics
- **Hallucination AUROC**: AUROC for hallucination detection using faithfulness
- **Relevance RMSE**: Root Mean Squared Error for context relevance predictions
- **Performance**: Mean query time, generation time, total time

### Output Files

```
logs/benchmark_ragbench/emanual/
├── 20251130_120000_run_config.json
├── 20251130_120000_raw_results.jsonl
├── 20251130_120000_detailed_results.jsonl  # ← With computed metrics
├── 20251130_120000_metrics.json             # ← Aggregate metrics
└── 20251130_120000_summary.csv              # ← Summary table
```

Example `metrics.json`:

```json
{
  "num_examples": 132,
  "mean_query_time": 0.234,
  "mean_generation_time": 1.456,
  "mean_total_time": 1.690,
  "ragas_faithfulness_mean": 0.847,
  "ragas_faithfulness_std": 0.123,
  "ragas_context_relevancy_mean": 0.782,
  "ragas_context_relevancy_std": 0.145,
  "trulens_groundedness_mean": 0.823,
  "trulens_groundedness_std": 0.156,
  "trulens_context_relevance_mean": 0.791,
  "trulens_context_relevance_std": 0.138,
  "hallucination_auroc_ragas": 0.892,
  "relevance_rmse_ragas": 0.145
}
```

---

## Supported RAGBench Datasets

The system supports all 12 RAGBench sub-datasets:

1. **emanual** - E-manuals (user manuals)
2. **covidqa** - COVID-19 QA
3. **cuad** - Contract understanding
4. **delucionqa** - Delucion QA
5. **expertqa** - Expert QA
6. **finqa** - Financial QA
7. **hagrid** - HAGRID dataset
8. **hotpotqa** - Multi-hop QA
9. **msmarco** - MS MARCO
10. **pubmedqa** - PubMed QA
11. **tatqa** - Table + text QA
12. **techqa** - Technical QA

To benchmark a different dataset, simply change the `--dataset` parameter:

```bash
# Example: Benchmark CovidQA
uv run python -m synth_rag.benchmark_ingest \
    --dataset covidqa \
    --split all \
    --recreate-collection

uv run python -m synth_rag.benchmark_runner \
    --dataset covidqa \
    --split test
```

---

## Best Practices

### 1. Start Small

Test with a small subset before running full benchmarks:

```bash
# Test with 10 examples first
uv run python -m synth_rag.benchmark_runner \
    --dataset emanual \
    --split test \
    --max-examples 10
```

### 2. Monitor API Usage

- TruLens and RAGAS both make OpenAI API calls for evaluation
- Use `--max-concurrent` to control rate limits
- Consider costs when evaluating large datasets

### 3. Incremental Evaluation

You can skip metrics you've already computed:

```bash
# Skip RAGAS if already computed
uv run python -m synth_rag.benchmark_metrics \
    --results-file logs/benchmark_ragbench/emanual/20251130_120000_raw_results.jsonl \
    --skip-ragas
```

### 4. Experiment with Parameters

Test different retrieval parameters:

```bash
# Higher top-k for more context
uv run python -m synth_rag.benchmark_runner \
    --dataset emanual \
    --split test \
    --top-k 10 \
    --prefetch-limit 100
```

---

## Troubleshooting

### Issue: Collection Not Found

```
Error: Collection ragbench_emanual does not exist
```

**Solution**: Run ingestion first:
```bash
uv run python -m synth_rag.benchmark_ingest --dataset emanual --split all
```

### Issue: RAGAS Evaluation Fails

```
Error: Rate limit exceeded
```

**Solution**: Add delays or reduce batch size in RAGAS evaluation code, or wait and retry.

### Issue: TruLens Timeout

```
Error: TruLens evaluation timeout
```

**Solution**: Reduce `--max-concurrent` parameter:
```bash
uv run python -m synth_rag.benchmark_metrics \
    --results-file ... \
    --max-concurrent 5
```

### Issue: Out of Memory

```
Error: CUDA out of memory
```

**Solution**: The benchmarking system doesn't use ColPali, so this shouldn't occur. If it does, reduce batch sizes in the embedding models.

---

## Comparing Results

To compare different configurations, run benchmarks with different parameters:

```bash
# Baseline: top-k=5
uv run python -m synth_rag.benchmark_runner \
    --dataset emanual \
    --split test \
    --top-k 5

# Experiment: top-k=10
uv run python -m synth_rag.benchmark_runner \
    --dataset emanual \
    --split test \
    --top-k 10
```

Then compare the `summary.csv` files from each run.

---

## Next Steps

- [Architecture](architecture.md) - Understand the system internals
- [Usage Guide](usage.md) - Learn about the main synth-rag features
- [API Reference](api/settings.md) - Explore the codebase

