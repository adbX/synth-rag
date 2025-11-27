# Quickstart

Get up and running with Synth-RAG in under 5 minutes.

---

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- API keys for Qdrant, OpenAI, and Brave Search

---

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/adbX/synth-rag.git
cd synth-rag
```

2. **Install dependencies**:
```bash
uv sync
```

3. **Create `.env` file**:
```bash
cat <<'EOF' > .env
QDRANT_URL="https://<your-qdrant-cluster>"
QDRANT_KEY="<your-api-key>"
OPENAI_API_KEY="<your-openai-key>"
BRAVE_API_KEY="<your-brave-key>"
EOF
```

---

## Verify Installation

Run these commands to verify core dependencies:

```bash
# Check PyTorch
uv run python -c "import torch; print('Torch:', torch.__version__)"

# Check ColPali
uv run python -c "from colpali_engine.models import ColPali; print('ColPali ready')"

# Check Qdrant client
uv run python -c "from qdrant_client import QdrantClient; print('Qdrant client ok')"
```

---

## Basic Usage

```mermaid
flowchart LR
    A[Install Dependencies] --> B[Configure API Keys]
    B --> C[Ingest Test Manuals]
    C --> D[Query Manuals]
    D --> E[Use Agent]
    
    style A fill:#e3f2fd
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#e8f5e9
    style E fill:#fce4ec
```

### Step 1: Ingest Test Manuals

Start with the test subset (3 PDFs):

```bash
uv run python -m synth_rag.manuals_ingest \
    --subset test \
    --collection midi_manuals \
    --device mps \
    --recreate-collection \
    --clear-tmp
```

!!! tip "Device Selection"
    - Use `--device mps` for Apple Silicon (M1/M2/M3)
    - Use `--device cuda:0` for NVIDIA GPUs
    - Use `--device cpu` as fallback

This will:

1. Render PDF pages to images
2. Extract text per page
3. Generate ColPali multivectors
4. Create dense/sparse embeddings
5. Upload to Qdrant

---

### Step 2: Query the Manuals

Ask a question using hybrid search:

```bash
uv run python -m synth_rag.manuals_query \
    --question "How do I adjust reverb settings on the Digitone II?" \
    --collection midi_manuals \
    --top-k 5 \
    --device mps
```

**Output**: Pretty-printed results with scores, manual names, page numbers, and text snippets.

---

### Step 3: Use the Agent

For more complex queries with web search fallback:

```bash
uv run python -m synth_rag.manuals_agent \
    --question "What are the key differences between Digitakt and Digitone?" \
    --collection midi_manuals \
    --model gpt-4o-mini \
    --device mps
```

The agent will:

1. Query your local manuals first
2. Search the web if needed
3. Generate a cited, grounded answer

---

## Next Steps

- [Full Setup Guide](setup.md) - Configure for production use
- [Usage Examples](usage.md) - Explore all features
- [Architecture](architecture.md) - Understand how it works

---

## Quick Troubleshooting

### Out of Memory?
```bash
# Reduce batch size
uv run python -m synth_rag.manuals_ingest --batch-size 2
```

### Collection Already Exists?
```bash
# Recreate from scratch
uv run python -m synth_rag.manuals_ingest --recreate-collection
```

### Slow Queries?
```bash
# Adjust prefetch limit
uv run python -m synth_rag.manuals_query --prefetch-limit 100 --top-k 3
```

See [Troubleshooting](troubleshooting.md) for more help.

