---
hide:
  - navigation
---

# Setup Guide

## System Requirements

- **Python**: 3.13 or higher
- **Package Manager**: [uv](https://docs.astral.sh/uv/)
- **GPU** (optional): Apple Silicon (MPS) or NVIDIA GPU (CUDA)

---

## Installation

### 1. Clone Repository

```bash
git clone https://github.com/adbX/synth-rag.git
cd synth-rag
```

### 2. Install Dependencies

```bash
uv sync
```

### 3. Configure API Keys

Create a `.env` file:

```ini
# Qdrant Vector Database
QDRANT_URL="https://xyz-example.eu-central.aws.cloud.qdrant.io:6333"
QDRANT_KEY="your-qdrant-api-key-here"

# OpenAI (for LLM)
OPENAI_API_KEY="sk-your-openai-key-here"

# Brave Search (for web search)
BRAVE_API_KEY="your-brave-search-key-here"
```

#### Getting API Keys

=== "Qdrant"
    Sign up at [cloud.qdrant.io](https://cloud.qdrant.io/)

=== "OpenAI"
    Visit [platform.openai.com](https://platform.openai.com/)

=== "Brave Search"
    Go to [brave.com/search/api](https://brave.com/search/api/)

---

## Directory Structure

```mermaid
graph TD
    Root[synth-rag/]
    
    Root --> Env[.env<br/>API keys - you create this]
    Root --> Pyproject[pyproject.toml<br/>Python dependencies]
    Root --> Lock[uv.lock<br/>Locked dependencies]
    
    Root --> Src[src/]
    Src --> SynthRag[synth_rag/]
    SynthRag --> Settings[settings.py]
    SynthRag --> Ingest[manuals_ingest.py]
    SynthRag --> Query[manuals_query.py]
    SynthRag --> Agent[manuals_agent.py]
    SynthRag --> UI[manuals_ui.py]
    
    Root --> Docs[documents/]
    Docs --> MIDI[midi_synthesizers/]
    MIDI --> Input[input/]
    Input --> Test[test/<br/>3 test PDFs]
    Input --> Full[full/<br/>8 full PDFs]
    MIDI --> Tmp[tmp/<br/>⚙️ auto-created]
    Tmp --> Pages[pages/<br/>rendered images]
    Tmp --> Text[text/<br/>extracted text]
    
    Root --> Logs[logs/<br/>⚙️ auto-created]
    Logs --> QueryLogs[manuals_queries/<br/>query logs]
    
    style Env fill:#fff3e0
    style Tmp fill:#e8f5e9
    style Logs fill:#e8f5e9
    style Settings fill:#e3f2fd
    style Ingest fill:#f3e5f5
    style Query fill:#f3e5f5
    style Agent fill:#f3e5f5
    style UI fill:#f3e5f5
```

---

## Device Configuration

### Apple Silicon (M1/M2/M3)

```bash
--device mps
```

### NVIDIA GPU

```bash
--device cuda:0
```

### CPU Only

```bash
--device cpu
```

---

## Model Downloads

```mermaid
graph LR
    subgraph Download[Auto-Download on First Run]
        ColPali[vidore/colpali-v1.3<br/>~2GB<br/>Vision-Language Model]
        MiniLM[sentence-transformers/<br/>all-MiniLM-L6-v2<br/>~90MB<br/>Dense Embeddings]
        BM25[Qdrant/bm25<br/><1MB<br/>Sparse Embeddings]
    end
    
    Cache[~/.cache/huggingface/]
    
    ColPali --> Cache
    MiniLM --> Cache
    BM25 --> Cache
    
    style ColPali fill:#ffccbc
    style MiniLM fill:#c5e1a5
    style BM25 fill:#b2ebf2
    style Cache fill:#fff9c4
```

**Model Summary:**

| Model | Size | Purpose |
|-------|------|---------|
| `vidore/colpali-v1.3` | ~2GB | ColPali vision-language model |
| `sentence-transformers/all-MiniLM-L6-v2` | ~90MB | Dense text embeddings |
| `Qdrant/bm25` | <1MB | Sparse keyword embeddings |

---

## Next Steps

- [Usage Guide](usage.md) - Learn how to use each component
- [Quickstart](quickstart.md) - Run your first queries
- [Architecture](architecture.md) - Understand the system design
