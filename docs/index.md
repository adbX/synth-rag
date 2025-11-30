---
hide:
  - navigation
---

# Synth-RAG

A retrieval-augmented generation (RAG) system for querying PDF manuals of MIDI synthesizers using ColPali multivector embeddings, hybrid search, and agentic workflows with LangGraph.

## Key Features

- **ColPali Multivector Embeddings** - Vision Language Models process PDF pages as images
- **Hybrid Search** - Combines dense (FastEmbed), sparse (BM25), and multivector representations
- **Two-Stage Retrieval** - Fast prefetch with HNSW-indexed vectors, precise reranking with ColPali
- **Agentic RAG** - LangGraph-powered agent with manual search and web fallback
- **Scalable** - Optimized for large PDF collections

## Quick Example

```bash
# Ingest manuals
uv run python -m synth_rag.manuals_ingest --subset test --collection midi_manuals

# Query with hybrid search
uv run python -m synth_rag.manuals_query \
    --question "How do I set up MIDI channels on the Digitone II?"

# Use agentic workflow
uv run python -m synth_rag.manuals_agent \
    --question "What are the differences between Digitakt and Digitone?"
```

---

## Technology Stack

```mermaid
graph TD
    subgraph Storage[Storage Layer]
        Qdrant[Qdrant<br/>Vector Database]
    end
    
    subgraph Embeddings[Embedding Models]
        ColPali[ColPali<br/>vidore/colpali-v1.3<br/>Vision-Language Model]
        FastEmbed[FastEmbed<br/>all-MiniLM-L6-v2<br/>Dense Embeddings]
        BM25[BM25<br/>Sparse Embeddings]
    end
    
    subgraph Agent[Agent Layer]
        LangGraph[LangGraph<br/>Agent Framework]
        GPT[OpenAI GPT-4o-mini<br/>LLM]
        Brave[Brave Search API<br/>Web Search]
    end
    
    subgraph Tools[Development]
        UV[uv<br/>Package Manager]
    end
    
    Embeddings --> Qdrant
    Agent --> Qdrant
    
    style Qdrant fill:#dc143c,color:#fff
    style ColPali fill:#4a90e2,color:#fff
    style LangGraph fill:#2ecc71,color:#fff
    style GPT fill:#9b59b6,color:#fff
```

---

## Next Steps

- [Quickstart Guide](quickstart.md) - Get up and running
- [Setup Instructions](setup.md) - Installation guide
- [Usage Examples](usage.md) - Learn how to use each component
- [Architecture](architecture.md) - Understand the system design
- [Benchmarking Guide](benchmarking.md) - Evaluate performance with RAGBench
- [API Reference](api/settings.md) - Explore the codebase
