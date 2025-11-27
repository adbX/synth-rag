# MIDI Manuals RAG System

A retrieval-augmented generation (RAG) system for querying PDF manuals of MIDI synthesizers using ColPali multivector embeddings, hybrid search, and agentic workflows with LangGraph.

## Features

- **ColPali Multivector Embeddings**: Uses Vision Language Models to process PDF pages directly as images
- **Hybrid Search**: Combines dense (FastEmbed), sparse (BM25), and multivector representations
- **Two-Stage Retrieval**: Fast first-stage retrieval with mean-pooled vectors, precise reranking with original multivectors
- **Agentic RAG**: LangGraph-powered agent that can query manuals and search the web
- **Scalable**: Optimized for large PDF collections with efficient indexing

## Quickstart

```
uv run python -m synth_rag.manuals_ingest --subset test --collection midi_manuals
uv run python -m synth_rag.manuals_query --question "Your question here"
uv run python -m synth_rag.manuals_agent --question "Your question here"
uv run synth-rag
```

## Setup

### Prerequisites

1. **Clone and navigate to the repository**:
   ```bash
   cd /Users/adb/stuff/gitclones/synth-rag
   ```

2. **Sync dependencies** (using `uv`):
   ```bash
   uv sync
   ```

3. **Create `.env` file** with required API keys:
   ```bash
   cat <<'EOF' > .env
   QDRANT_URL="https://<your-qdrant-cluster>"
   QDRANT_KEY="<your-api-key>"
   OPENAI_API_KEY="<your-openai-key>"
   BRAVE_API_KEY="<your-brave-key>"
   EOF
   ```

4. **Verify core libraries**:
   ```bash
   uv run python -c "import torch; print('Torch:', torch.__version__)"
   uv run python -c "from colpali_engine.models import ColPali; print('ColPali ready')"
   uv run python -c "from qdrant_client import QdrantClient; print('Qdrant client ok')"
   ```

### Directory Structure

The system uses the following directories:

- `documents/midi_synthesizers/input/test/` - Test subset of PDF manuals
- `documents/midi_synthesizers/input/full/` - Full collection of PDF manuals
- `documents/midi_synthesizers/tmp/pages/` - Rendered page images (auto-created)
- `documents/midi_synthesizers/tmp/text/` - Extracted text manifests (auto-created)
- `logs/manuals_queries/` - Query logs (auto-created)

## Usage

### 1. Ingest PDF Manuals

Process PDF manuals and index them in Qdrant:

```bash
# Ingest test subset (recommended first)
uv run python -m synth_rag.manuals_ingest \
    --subset test \
    --collection midi_manuals \
    --device mps \
    --recreate-collection \
    --clear-tmp

# Ingest full collection
uv run python -m synth_rag.manuals_ingest \
    --subset full \
    --collection midi_manuals \
    --device mps
```

**Options**:
- `--subset {test,full}`: Which subset to ingest
- `--collection NAME`: Qdrant collection name (default: `midi_manuals`)
- `--device {mps,cuda:0,cpu}`: Device for ColPali model
- `--batch-size N`: Batch size for embedding (default: 4)
- `--clear-tmp`: Clear temporary directories before ingestion
- `--recreate-collection`: Delete and recreate the Qdrant collection

**What it does**:
1. Renders PDF pages to RGB images using `pypdfium2`
2. Extracts per-page text using `pymupdf`
3. Generates ColPali multivectors (original + mean-pooled rows/cols)
4. Creates dense (FastEmbed) and sparse (BM25) embeddings for text chunks
5. Upserts all embeddings to Qdrant with metadata

### 2. Query Manuals

Query the indexed manuals using hybrid search:

```bash
uv run python -m synth_rag.manuals_query \
    --question "In the FM Drum Machine of Digitone II, what are all the ways to increase the decay time?" \
    --collection midi_manuals \
    --top-k 5 \
    --device mps
```

**Options**:
- `--question TEXT`: Question to ask (required)
- `--collection NAME`: Qdrant collection name
- `--top-k N`: Number of results to return (default: 5)
- `--prefetch-limit N`: Results to prefetch for reranking (default: 50)
- `--device {mps,cuda:0,cpu}`: Device for ColPali model
- `--manual-filter NAME`: Filter by specific manual name (optional)

**Output**:
- Pretty-printed results with scores, manual names, page numbers, and text snippets
- Query logs saved to `logs/manuals_queries/<timestamp>.json`

### 3. Agentic RAG

Use the LangGraph agent for interactive Q&A with web search fallback:

```bash
uv run python -m synth_rag.manuals_agent \
    --question "What are the differences between the Digitakt and Digitone?" \
    --collection midi_manuals \
    --model gpt-4o-mini \
    --device mps
```

**Options**:
- `--question TEXT`: Question to ask (required)
- `--collection NAME`: Qdrant collection name
- `--model NAME`: OpenAI model (default: `gpt-4o-mini`)
- `--device {mps,cuda:0,cpu}`: Device for ColPali model
- `--top-k N`: Results per retrieval (default: 3)

**What it does**:
1. Agent analyzes the question
2. Decides whether to use manual retrieval tool or web search tool
3. Retrieves relevant information
4. Generates a grounded answer with citations

## Architecture

### Collection Schema

The Qdrant collection uses multiple vector types:

- **`colpali_original`** (128-dim multivector): Original ColPali embeddings, no HNSW index (reranking only)
- **`colpali_rows`** (128-dim multivector): Mean-pooled by rows, HNSW indexed for fast retrieval
- **`colpali_cols`** (128-dim multivector): Mean-pooled by columns, HNSW indexed for fast retrieval
- **`dense`** (384-dim): FastEmbed dense embeddings (all-MiniLM-L6-v2)
- **`sparse`**: BM25 sparse embeddings with IDF modifier

### Two-Stage Retrieval

1. **First Stage**: Fast prefetch using HNSW-indexed vectors (`dense`, `sparse`, `colpali_rows`, `colpali_cols`)
2. **Second Stage**: Precise reranking using original ColPali multivectors

This approach provides:
- **10x faster indexing** compared to indexing original multivectors
- **Comparable retrieval quality** to using original vectors directly
- **Scalability** to large PDF collections (20,000+ pages)

## Maintenance

### Clean Temporary Directories

```bash
rm -rf documents/midi_synthesizers/tmp/pages/*
rm -rf documents/midi_synthesizers/tmp/text/*
```

Or use the `--clear-tmp` flag when running ingestion.

### Monitor Collection Size

```python
from synth_rag.settings import get_qdrant_client

client = get_qdrant_client()
info = client.get_collection("midi_manuals")
print(f"Points: {info.points_count}")
print(f"Vectors: {info.vectors_count}")
```

### Delete Collection

```bash
uv run python -c "
from synth_rag.settings import get_qdrant_client
client = get_qdrant_client()
client.delete_collection('midi_manuals')
print('Collection deleted')
"
```

## Technical Details

### Why Extract Text with PyMuPDF?

While ColPali processes PDFs as images, we still extract text for:
1. **Text-based embeddings**: Dense and sparse vectors for hybrid search
2. **Payload metadata**: Searchable text snippets in results
3. **Context snippets**: Human-readable text for the agent and users

### Device Selection

- **`mps`**: Apple Silicon (M1/M2/M3) - recommended for macOS
- **`cuda:0`**: NVIDIA GPU - fastest for large batches
- **`cpu`**: CPU fallback - slowest but works everywhere

### Model Details

- **ColPali**: `vidore/colpali-v1.3` - generates ~1,030 vectors per page (32×32 patches + special tokens)
- **Dense**: `sentence-transformers/all-MiniLM-L6-v2` - 384-dim embeddings
- **Sparse**: `Qdrant/bm25` - keyword-based retrieval
- **LLM**: OpenAI `gpt-4o-mini` (configurable)

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
uv run python -m synth_rag.manuals_ingest --batch-size 2
```

### Slow Queries

Increase prefetch limit or reduce top-k:
```bash
uv run python -m synth_rag.manuals_query --prefetch-limit 100 --top-k 3
```

### Collection Already Exists

Use `--recreate-collection` to start fresh:
```bash
uv run python -m synth_rag.manuals_ingest --recreate-collection
```