<!-- 6ba21ce0-5a5e-42fd-98a0-3f504a33315f f983c3a0-30aa-4731-a49d-9c2b2d4f8747 -->
# Paperqa Computo Query Script

## Overview

Build an async Python script in `paperqa_computo.py` that queries Computo papers using paperqa, with two modes: single paper query or batch query all papers. Results saved as organized JSON logs.

## Implementation Steps

### 1. Core Functionality

- **Read computo_repos.csv** to get list of papers and construct .qmd paths
- **Find and combine .qmd files**: For each paper, find main .qmd (matches repo name), append any supplementary .qmd files (e.g., `-supp.qmd`)
- **Async query function**: Use `paperqa.Docs()` and `aadd()` / `aquery()` for each paper independently
- **Progress tracking**: Use `tqdm.asyncio` for progress bars when querying multiple papers

### 2. CLI Interface (using argparse)

Arguments:

- `--question` (required): The question to ask
- `--paper` (optional): Specific paper name from CSV to query (if omitted, queries all)
- `--model` (required): LLM model for paperqa (e.g., `claude-3-5-sonnet-20240620`)
- `--output-dir` (optional): Directory for logs (default: `logs/paperqa`)
- `--max-sources` (optional): Max sources for answer (default: 3)

### 3. Logging Structure

Organize logs in sub-folders:

```
logs/paperqa/
  {timestamp}/
    query_metadata.json  # Overall query info
    {paper_name}_response.json  # Individual paper response
```

Each JSON log contains:

- `question`: The query asked
- `paper_name`: Paper identifier
- `model`: LLM model used
- `settings`: paperqa Settings dict
- `timestamp`: ISO format timestamp
- `response_time_seconds`: Measured query time
- `status`: Status value from response
- `answer`: The answer text
- `contexts`: List of context objects (source, content, score)
- `session_data`: Full session object serialized

### 4. File Operations

- Use `pathlib.Path` throughout
- CSV reading with pandas (already in dependencies)
- For .qmd combination: read files as text, concatenate with separator comments

### 5. Error Handling

- Graceful handling if .qmd files missing
- Continue on individual paper failures when querying all
- Log errors to separate error log file

## Key Files

- `/Users/adb/stuff/gitclones/qdrant-init/src/qdrant_init/paperqa_computo.py` - Main script
- `/Users/adb/stuff/gitclones/qdrant-init/documents/computo/computo_repos.csv` - Paper list

## Notes

- paperqa doesn't officially support .qmd but we'll try it as .md equivalent
- Each document queried independently (no shared Docs object across papers)
- Async execution for efficiency with proper progress feedback

### To-dos

- [ ] Set up imports and CLI argument parsing with argparse
- [ ] Implement functions to read CSV and discover/combine .qmd files per paper
- [ ] Implement async function to query a single paper with paperqa
- [ ] Implement async function to query all papers with progress tracking
- [ ] Implement JSON logging system with organized subfolder structure
- [ ] Create main async function to orchestrate single vs batch mode