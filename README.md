# MIDI Manuals RAG System

A retrieval-augmented generation (RAG) system for querying PDF manuals of MIDI synthesizers using ColPali multivector embeddings, hybrid search, and agentic workflows with LangGraph.

## Features

- **ColPali Multivector Embeddings**: Uses Vision Language Models to process PDF pages directly as images
- **Hybrid Search**: Combines dense (FastEmbed), sparse (BM25), and multivector representations
- **Two-Stage Retrieval**: Fast first-stage retrieval with mean-pooled vectors, precise reranking with original multivectors
- **Agentic RAG**: LangGraph-powered agent that can query manuals and search the web
- **Scalable**: Optimized for large PDF collections with efficient indexing

## Setup

### Prerequisites

1. **Clone and navigate to the repository**:
   ```bash
   cd /Users/adb/stuff/gitclones/qdrant-init
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
uv run python -m qdrant_init.manuals_ingest \
    --subset test \
    --collection midi_manuals \
    --device mps \
    --recreate-collection \
    --clear-tmp

# Ingest full collection
uv run python -m qdrant_init.manuals_ingest \
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
uv run python -m qdrant_init.manuals_query \
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
uv run python -m qdrant_init.manuals_agent \
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
from qdrant_init.settings import get_qdrant_client

client = get_qdrant_client()
info = client.get_collection("midi_manuals")
print(f"Points: {info.points_count}")
print(f"Vectors: {info.vectors_count}")
```

### Delete Collection

```bash
uv run python -c "
from qdrant_init.settings import get_qdrant_client
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
uv run python -m qdrant_init.manuals_ingest --batch-size 2
```

### Slow Queries

Increase prefetch limit or reduce top-k:
```bash
uv run python -m qdrant_init.manuals_query --prefetch-limit 100 --top-k 3
```

### Collection Already Exists

Use `--recreate-collection` to start fresh:
```bash
uv run python -m qdrant_init.manuals_ingest --recreate-collection
```

## clone_computo_repos.py

Your role is to write a minimal python script to clone the repos of all published papers in the COMPUTO organization. Published repo names begin with the "published" and followed by the date of publication like so "published-202510-durand-fast".
organization url: https://github.com/orgs/computorg/repositories?type=source
sample repo names:
```
published-202510-durand-fast
regMMD-paper
published-202412-ambroise-spectral
published-202306-sanou-multiscale_glasso
published-202311-delattre-fim
template-computo-julia
workflows
comm
published-202407-susmann-adaptive-conformal
jds2024-workshop
```

The script `clone_computo_repos.py` should:
- Clone all repos into `documents/computo/` using PyGithub (https://pygithub.readthedocs.io/en/stable/introduction.html#very-short-tutorial). 
- Have a progress bar using `tqdm` to track the progress of the cloning process and print the number of repos cloned so far out of the total found.
- Create directories as necessary and use `pathlib` for file operations.
- Create a csv file `documents/computo/computo_repos.csv` with the following columns: repo_name, repo_url, year_of_publication and metadata. Metadata should include all the tags after the year and date. For example: `published-202510-durand-fast` should have the metadata `durand-fast`. And `published-202306-sanou-multiscale_glasso` should have the metadata `sanou-multiscale_glasso`.
- Do not re-clone repos that are already cloned by checking the csv file.

## paperqa

Create a minimal python script to use `paperqa` (https://edisonscientific.gitbook.io/edison-cookbook/paperqa#manual-no-agent-adding-querying-documents) to query the main `.qmd` document in each repo in `documents/computo/` and return the answer to a given question (all papers in available are listed in `documents/computo/computo_repos.csv` which can be used to get the .qmd path without searching). I am querying each document individually to test for reproducibility so do not create a *set* of documents for reference, the queries for one document shoudl be independent of other documents.

`paperqa` does not support .qmd officially, but it does support .md files so just try to use .qmd instead without any conversion.
```
from paperqa import Docs, Settings

# valid extensions include .pdf, .txt, .md, .html, .docx, .xlsx, .pptx, and code files (e.g., .py, .ts, .yaml)
doc_paths = ("myfile.pdf", "myotherfile.pdf")

# Prepare the Docs object by adding a bunch of documents
docs = Docs()
for doc_path in doc_paths:
    await docs.aadd(doc_path)

# Set up how we want to query the Docs object
settings = Settings()
settings.llm = "claude-3-5-sonnet-20240620"
settings.answer.answer_max_sources = 3

# Query the Docs object to get an answer
session = await docs.aquery("What is PaperQA2?", settings=settings)
print(session)
```

`paperqa` returns a `PQASession` object with the following attributes:
```
print("Let's examine the PQASession object returned by paperqa:\n")

print(f"Status: {response.status.value}")

print("1. Question asked:")
print(f"{response.session.question}\n")

print("2. Answer provided:")
print(f"{response.session.answer}\n")

print("4. Contexts used to generate the answer:")
print(
    "These are the relevant text passages that were retrieved and used to formulate the answer:"
)
for i, ctx in enumerate(response.session.contexts, 1):
    print(f"\nContext {i}:")
    print(f"Source: {ctx.text.name}")
    print(f"Content: {ctx.context}")
    print(f"Score: {ctx.score}")
```

For a single question, the script should be able to:
- Query one question against one specified paper
- Query one question against all papers in COMPUTO and return a dataframe with the question, answer, response time, and timestamp.

I want a robust logger/saver to record all data for queries which should at least include the question, session object, the measured response time, run settings, timestamp, and any other relevant data. The logs should be organized well and easily queryable. Use pathlib for file operations. Make sure the code is simple and does not rely on too many external dependencies.

1. b) Make it configurable via CLI argument (also make sure to save this in the logs)
2. a) One per query is fine but use sub-folders for nicer organization
3. b) If there are multiple .qmd files append the supplementary ones to the main file.
4. a) use async. also add progress tracking for it.

python src/qdrant_init/paperqa_computo.py \
    --question "What is the main contribution of this paper?" \
    --paper "published-202301-boulin-clayton" \
    --model "claude-3-5-sonnet-20240620"

    python src/qdrant_init/paperqa_computo.py \
    --question "What methodology does this paper use?" \
    --model "claude-3-5-sonnet-20240620"

## synthesizer manual q/a with docling vlm, qdrant and llamaindex

### create docs

My goal is to create a chatbot that can answer questions about PDF manuals of midi synthesizers. I have the manuals in the `documents/midi_synthesizers/input/test/` (a subset to test) and `documents/midi_synthesizers/input/full/` (all manuals) folders. Create temp directories for converted/parsed manuals and use them for indexing and retrieval. I want to use `qdrant` for the vector database. The remaining tech stack and libraries are up to you so decide what is best for the tasks described below. I don't need all the libraries and methods described in the qdrant documentation, just the ones that are necessary to achieve the tasks described below. First decide what is the best approach to achieve the tasks described below and then implement it. All relevant documentation and notebooks for `qdrant` are in the `documents/qdrant-docs/` folder.

First read all documentation and notebooks in the `documents/qdrant-docs/` folder and create a file with high level summaries of each file the method/tech stack being used and what the best use case for each component is. I want to mix and match the components to achieve the best results.

### write scripts

My goal is to create a chatbot that can answer questions about PDF manuals of midi synthesizers. I have the manuals in the `documents/midi_synthesizers/input/test/` (a subset to test) and `documents/midi_synthesizers/input/full/` (all manuals) folders. Create temp directories for converted/parsed manuals and use them for indexing and retrieval. I want to use `qdrant` for the vector database. The remaining tech stack and libraries are up to you so decide what is best for the tasks described below. I don't need all the libraries and methods described in the qdrant documentation, just the ones that are necessary to achieve the tasks described below. All relevant documentation and notebooks for `qdrant` are in the `documents/qdrant-docs/` folder.`documents/qdrant-docs/qdrant_component_summaries.md` is a machine-readable table that captures each doc/notebook’s tech stack, core method, and ideal use cases so you can mix components as needed.

- Use the `colpali` library to create multivector representations of the pdf manuals instead of the `docling` library `documents/qdrant-docs/pdf-retrieval-at-scale.md`.
- Use hybrid search with reranking `documents/qdrant-docs/reranking-hybrid-search.md`
- Use an agentic RAG approach `documents/qdrant-docs/agentic-rag-langgraph.md`. I don't have 2 vector databases (just use one). But use the `brave_search` library to search the web. I have the api key in the .env file.
- API keys in my `.env` file: QDRANT_KEY, QDRANT_URL, BRAVE_API_KEY, OPENAI_API_KEY
- Use `pathlib` for file operations. I am using `uv` (https://docs.astral.sh/uv/getting-started/features/) for package management and running scripts. Create modular, minimal python scripts with a command line interface to acheive my tasks. 

1. Don't ever manually modify packages in pyproject.toml. Only add/remove using uv commands like uv add, uv remove, etc. I have already added all the new requirements so skip the step this time.
2. Isnt the colpali model directly parsing pdf pages into multivector embeddings? What is the point of extracting per page text using pymupdf?

Modify my plan to give me a list of all the setup commands/initial config I need to do in the terminal, don't run them yourself. I will provide the output which you can use to build the app.

uv run python -c "import torch; print('Torch:', torch.__version__)"
Torch: 2.8.0

uv run python -c "from colpali_engine.models import ColPali; print('ColPali ready')"
ColPali ready

❯ uv run python -c "from qdrant_client import QdrantClient; print('Qdrant client ok')"
Qdrant client ok