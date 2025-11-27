# Settings Module

Configuration and utility functions for Synth-RAG.

---

## Overview

The `settings` module provides:

- **API configuration** - Loads credentials from `.env` file
- **Directory paths** - Centralized path management
- **Client factories** - Singleton Qdrant client
- **Utility functions** - Path validation, directory creation

---

## Module Reference

::: synth_rag.settings
    options:
      show_source: true
      members:
        - APISettings
        - get_api_settings
        - get_qdrant_client
        - get_manual_input_dir
        - ensure_tmp_dirs
        - ensure_logs_dir
        - REPO_ROOT
        - DOCS_DIR
        - MIDI_DIR
        - INPUT_DIR
        - TMP_DIR
        - TMP_PAGES_DIR
        - TMP_TEXT_DIR
        - LOGS_DIR
        - MANUAL_QUERY_LOGS_DIR

---

## Usage Examples

### Load API Settings

```python
from synth_rag.settings import get_api_settings

settings = get_api_settings()
print(settings.qdrant_url)
print(settings.openai_key)
```

### Get Qdrant Client

```python
from synth_rag.settings import get_qdrant_client

client = get_qdrant_client()
collections = client.get_collections()
```

### Work with Directories

```python
from synth_rag.settings import get_manual_input_dir, ensure_tmp_dirs

# Get input directory for test subset
test_dir = get_manual_input_dir("test")

# Ensure tmp directories exist
pages_dir, text_dir = ensure_tmp_dirs(clear=True)
```

---

## Environment Variables

```mermaid
graph TD
    EnvFile[.env file]
    
    EnvFile --> Qdrant[QDRANT_URL<br/>QDRANT_KEY]
    EnvFile --> OpenAI[OPENAI_API_KEY]
    EnvFile --> Brave[BRAVE_API_KEY]
    
    Qdrant --> VectorDB[Vector Database<br/>Storage & Retrieval]
    OpenAI --> LLM[LLM<br/>Agent Reasoning]
    Brave --> WebSearch[Web Search<br/>Fallback Tool]
    
    style EnvFile fill:#fff3e0
    style Qdrant fill:#dc143c,color:#fff
    style OpenAI fill:#9b59b6,color:#fff
    style Brave fill:#ff6b35,color:#fff
    style VectorDB fill:#f5f5f5
    style LLM fill:#f5f5f5
    style WebSearch fill:#f5f5f5
```

**Required Variables:**

| Variable | Description | Example |
|----------|-------------|---------|
| `QDRANT_URL` | Qdrant cluster URL | `https://xyz.aws.cloud.qdrant.io:6333` |
| `QDRANT_KEY` | Qdrant API key | `your-api-key-here` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `BRAVE_API_KEY` | Brave Search API key | `BSA...` |

---

## Directory Structure

```mermaid
graph TD
    Root[synth-rag/]
    
    Root --> Docs[documents/<br/>DOCS_DIR]
    Docs --> MIDI[midi_synthesizers/<br/>MIDI_DIR]
    
    MIDI --> Input[input/<br/>INPUT_DIR]
    Input --> Test[test/<br/>Test subset]
    Input --> Full[full/<br/>Full collection]
    
    MIDI --> Tmp[tmp/<br/>TMP_DIR]
    Tmp --> Pages[pages/<br/>TMP_PAGES_DIR<br/>rendered images]
    Tmp --> Text[text/<br/>TMP_TEXT_DIR<br/>extracted text]
    
    Root --> Logs[logs/<br/>LOGS_DIR]
    Logs --> QueryLogs[manuals_queries/<br/>MANUAL_QUERY_LOGS_DIR]
    
    style Root fill:#e3f2fd
    style Docs fill:#f3e5f5
    style MIDI fill:#fff3e0
    style Input fill:#e8f5e9
    style Tmp fill:#ffecb3
    style Logs fill:#ffe0b2
```
