# Quickstart

## Prerequisites

- Python 3.13+
- [uv](https://docs.astral.sh/uv/) package manager
- API keys for Qdrant, OpenAI, and Brave Search

---

## Installation

**1. Clone and install:**

```bash
git clone https://github.com/adbX/synth-rag.git
cd synth-rag
uv sync
```

**2. Create `.env` file:**

```bash
cat <<'EOF' > .env
QDRANT_URL="https://<your-qdrant-cluster>"
QDRANT_KEY="<your-api-key>"
OPENAI_API_KEY="<your-openai-key>"
BRAVE_API_KEY="<your-brave-key>"
EOF
```

**3. Verify installation:**

```bash
uv run python -c "import torch; print('Torch:', torch.__version__)"
uv run python -c "from colpali_engine.models import ColPali; print('ColPali ready')"
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

```bash
uv run python -m synth_rag.manuals_ingest \
    --subset test \
    --collection midi_manuals \
    --device mps \
    --recreate-collection \
    --clear-tmp
```

!!! tip "Device Selection"
    - `--device mps` for Apple Silicon (M1/M2/M3)
    - `--device cuda:0` for NVIDIA GPUs
    - `--device cpu` as fallback

### Step 2: Query the Manuals

```bash
uv run python -m synth_rag.manuals_query \
    --question "How do I adjust reverb settings on the Digitone II?" \
    --collection midi_manuals \
    --top-k 5 \
    --device mps
```

### Step 3: Use the Agent

```bash
uv run python -m synth_rag.manuals_agent \
    --question "What are the key differences between Digitakt and Digitone?" \
    --collection midi_manuals \
    --model gpt-4o-mini \
    --device mps
```

---

## Next Steps

- [Setup Guide](setup.md) - Detailed configuration
- [Usage Guide](usage.md) - Explore all features
- [Architecture](architecture.md) - Understand the system
