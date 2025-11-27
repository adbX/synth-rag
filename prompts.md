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

python src/synth_rag/paperqa_computo.py \
    --question "What is the main contribution of this paper?" \
    --paper "published-202301-boulin-clayton" \
    --model "claude-3-5-sonnet-20240620"

    python src/synth_rag/paperqa_computo.py \
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

Modify my code to use the docling vlm to donvert my pdf manuals for qdrant indexing.
- use granite-docling-258M-mlx as the model

docling --pipeline vlm --vlm-model granite_docling documents/midi_synthesizers/input/test/Digitone-2-User-Manual_ENG_OS1.10D_251022.pdf --to html

use the documentation below
```
VLM pipeline with GraniteDocling

Minimal VLM pipeline example: convert a PDF using a vision-language model.

What this example does

    Runs the VLM-powered pipeline on a PDF (by URL) and prints Markdown output.
    Shows two setups: default (Transformers/GraniteDocling) and macOS MPS/MLX.

Prerequisites

    Install Docling with VLM extras and the appropriate backend (Transformers or MLX).
    Ensure your environment can download model weights (e.g., from Hugging Face).

How to run

    From the repository root, run: python docs/examples/minimal_vlm_pipeline.py.
    The script prints the converted Markdown to stdout.

Notes

    source may be a local path or a URL to a PDF.
    The second section demonstrates macOS MPS acceleration via MLX (vlm_model_specs.GRANITEDOCLING_MLX).
    For more configurations and model comparisons, see docs/examples/compare_vlm_models.py.

from docling.datamodel import vlm_model_specs
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    VlmPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline

# Convert a public arXiv PDF; replace with a local path if preferred.
source = "https://arxiv.org/pdf/2501.17887"

###### USING SIMPLE DEFAULT VALUES
# - GraniteDocling model
# - Using the transformers framework

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
        ),
    }
)

doc = converter.convert(source=source).document

print(doc.export_to_markdown())


###### USING MACOS MPS ACCELERATOR
# Demonstrates using MLX on macOS with MPS acceleration (macOS only).
# For more options see the `compare_vlm_models.py` example.

pipeline_options = VlmPipelineOptions(
    vlm_options=vlm_model_specs.GRANITEDOCLING_MLX,
)

converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        ),
    }
)

doc = converter.convert(source=source).document

print(doc.export_to_markdown())
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

python src/synth_rag/paperqa_computo.py \
    --question "What is the main contribution of this paper?" \
    --paper "published-202301-boulin-clayton" \
    --model "claude-3-5-sonnet-20240620"

    python src/synth_rag/paperqa_computo.py \
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

# frontend gradio

Your role is to build a frontend for my RAG chatbot using gradio. The readme of my project is in the `README.md` file. The document for gradio relevant documentation to build chatbots with gradio is in the `documents/ui-docs/gradio.md` file. Be as simple and minimal as possible.

# add references and prioritize pdfs

Modify my chatbot to add references to page numbers for each part of answer. Also make sure to priorize answers from the the documents/manuals first and then add the web search results at the end in a separate section. The readme of my project is in the `README.md` file.

# fix missing refs

The chatbot does not have a section for the local documents/manuals at all. I am only getting web search results as seen in the conversation below. Help me fix this: 

```
user: How can I setup encoders of the Faderfox EC-4 MIDI controller such that encoders 2, 3 & 4 control the Reverb of the Digitone II on MIDI channels 6, 7, & 8 respectively? Then, I also want to use encoder 1 to control encoders 2, 3 & 4 simultaneously on the faderfox

assistant: ## Information from Web Search

### Faderfox EC-4 Setup
The Faderfox EC-4 is a MIDI controller with 16 gridless push encoders. It allows for the creation of up to 16 individual groups, enabling control of up to 256 parameters per group. The EC-4 features a "Link function" that can link up to 16 encoders or pushbuttons for multiple commands, which would be useful for your requirement to control multiple encoders simultaneously (Faderfox EC4, Perfect Circuit).

To set up the encoders:
1. **Choose the Encoder Setup**: You can select the encoder setup and/or group by pressing one of the grey SETUP or GROUP keys. The display will show 16 setup or group names in a 4x4 grid (micromodul EC4 User manual).

2. **Link Function**: Use the link function to link encoders 2, 3, and 4 to encoder 1. This will allow encoder 1 to control the other three encoders simultaneously.

3. **MIDI Channel Assignment**: Assign MIDI channels to the encoders. You can set encoders 2, 3, and 4 to control the Reverb on the Digitone II by assigning them to MIDI channels 6, 7, and 8 respectively.

### Digitone II MIDI Control
Unfortunately, I couldn't retrieve specific information about setting up MIDI control on the Digitone II from the web search. However, typically, you would need to ensure that the Digitone II is set to receive MIDI CC messages on the appropriate channels and that the parameters you wish to control (Reverb in this case) are mapped to the corresponding MIDI CC numbers.

For detailed setup, you would typically refer to the Digitone II's MIDI implementation chart in its manual to find the correct CC numbers for Reverb control and ensure the device is set to receive on the specified channels.

If you have access to the manuals, they would provide the most accurate and detailed instructions for both devices.
```

# rename

I have changed the name of my project to `synth-rag`. Give me a list of all the files/modules/commands/etc. that I need to rename to match the new name.

# mkdocs

I have mkdocs running with uv (`uv run mkdocs serve`). Create a comprehensive set of documentation using my project readme `README.md` and the scripts in the `src/synth_rag/` folder. Update the mkdocs.yml file to reflect the new name and structure. I also have mkdocstrings plugin (https://mkdocstrings.github.io/python/usage/) for python docstrings. Use it to document the code in the `src/synth_rag/` folder.

# mermaidjs

For my mkdocs documentation, I want to use mermaidjs to make diagrams, flowcharts, tables, etc. Use the documentation in `documents/mkdocs-docs/mermaidjs.md` for reference. Update all the documentation in `/docs/` to use the mermaidjs where applicable.

## Summary

I've successfully updated your mkdocs documentation to use mermaidjs for diagrams, flowcharts, and visual representations. Here's what was done:

### 1. **Configuration** (`mkdocs.yml`)
- Added mermaidjs support to the `pymdownx.superfences` extension

### 2. **Architecture** (`docs/architecture.md`)
- ✅ System overview flowchart (replaces ASCII art)
- ✅ ColPali variants comparison diagram
- ✅ Agentic layer state diagram
- ✅ Indexing performance gantt chart
- ✅ Query performance comparison
- ✅ Scalability visualization
- ✅ Hybrid search advantages flowchart

### 3. **Index Page** (`docs/index.md`)
- ✅ Technology stack diagram (replaces table)

### 4. **Quickstart** (`docs/quickstart.md`)
- ✅ Basic usage workflow flowchart

### 5. **Setup Guide** (`docs/setup.md`)
- ✅ Directory structure tree diagram
- ✅ Model downloads visualization

### 6. **Usage Guide** (`docs/usage.md`)
- ✅ Ingestion pipeline flowchart
- ✅ Query output format structure
- ✅ First-time setup workflow
- ✅ Agent workflow (already had mermaid, kept it)

### 7. **Troubleshooting** (`docs/troubleshooting.md`)
- ✅ Quick diagnostic flowchart
- ✅ Memory issue decision tree

### 8. **API Reference - Agent** (`docs/api/manuals_agent.md`)
- ✅ LangGraph flow diagram (replaces ASCII art)
- ✅ Agent execution sequence diagram
- ✅ Model selection comparison
- ✅ Query timing gantt chart

### 9. **API Reference - Ingest** (`docs/api/manuals_ingest.md`)
- ✅ Pipeline flowchart with timing

### 10. **API Reference - Query** (`docs/api/manuals_query.md`)
- ✅ Search pipeline sequence diagram
- ✅ Scaling visualization

### 11. **API Reference - Settings** (`docs/api/settings.md`)
- ✅ Directory structure tree
- ✅ Environment variables flow

### 12. **API Reference - UI** (`docs/api/manuals_ui.md`)
- ✅ Chat interface flowchart

All diagrams use appropriate mermaid diagram types:
- **Flowcharts** for processes and workflows
- **Sequence diagrams** for interactions
- **State diagrams** for state machines
- **Gantt charts** for timing visualizations
- **Graph diagrams** for relationships and structures

The documentation now has a more professional, visual appearance while maintaining all the technical accuracy. The build completes successfully! 🎉

# ci mkdocs uv

I am using uv for package management and to use my repo as a package itself that mkdocstrings can use to document the code. I have a ci workflow that builds the docs and deploys them to the gh-pages branch. I want to use uv to build the docs and deploy them to the gh-pages branch. I am currently installing all the packages using uv sync in the ci workflow which is unnecessary if I am only building the docs. Can I use dependency groups to only install the packages needed for the docs? (https://docs.astral.sh/uv/concepts/projects/dependencies/#dependency-groups). Update the ci file and and list the uv commands I need to run for the groups (but don't run the latter) @.github/workflows/ci.yml @pyproject.toml 

In my `Build docs to gh-pages branch` it still seems to be installing all the packages for `synth-rag`. @.github/workflows/ci.yml How can I avoid doing this for my github pages documentation? Here are the logs for the run:
```
2025-11-27T21:34:25.5992707Z Current runner version: '2.329.0'
2025-11-27T21:34:25.6028770Z ##[group]Runner Image Provisioner
2025-11-27T21:34:25.6030177Z Hosted Compute Agent
2025-11-27T21:34:25.6031072Z Version: 20251016.436
2025-11-27T21:34:25.6032456Z Commit: 8ab8ac8bfd662a3739dab9fe09456aba92132568
2025-11-27T21:34:25.6033664Z Build Date: 2025-10-15T20:44:12Z
2025-11-27T21:34:25.6034647Z ##[endgroup]
2025-11-27T21:34:25.6035689Z ##[group]Operating System
2025-11-27T21:34:25.6036649Z Ubuntu
2025-11-27T21:34:25.6037445Z 24.04.3
2025-11-27T21:34:25.6038298Z LTS
2025-11-27T21:34:25.6039048Z ##[endgroup]
2025-11-27T21:34:25.6039787Z ##[group]Runner Image
2025-11-27T21:34:25.6040845Z Image: ubuntu-24.04
2025-11-27T21:34:25.6041664Z Version: 20251112.124.1
2025-11-27T21:34:25.6043827Z Included Software: https://github.com/actions/runner-images/blob/ubuntu24/20251112.124/images/ubuntu/Ubuntu2404-Readme.md
2025-11-27T21:34:25.6046697Z Image Release: https://github.com/actions/runner-images/releases/tag/ubuntu24%2F20251112.124
2025-11-27T21:34:25.6048472Z ##[endgroup]
2025-11-27T21:34:25.6050430Z ##[group]GITHUB_TOKEN Permissions
2025-11-27T21:34:25.6053145Z Contents: write
2025-11-27T21:34:25.6054149Z Metadata: read
2025-11-27T21:34:25.6055022Z ##[endgroup]
2025-11-27T21:34:25.6058025Z Secret source: Actions
2025-11-27T21:34:25.6059301Z Prepare workflow directory
2025-11-27T21:34:25.6773440Z Prepare all required actions
2025-11-27T21:34:25.6834302Z Getting action download info
2025-11-27T21:34:26.1993043Z Download action repository 'actions/checkout@v4' (SHA:34e114876b0b11c390a56381ad16ebd13914f8d5)
2025-11-27T21:34:26.7350727Z Download action repository 'actions/setup-python@v5' (SHA:a26af69be951a213d495a4c3e4e4022e16d87065)
2025-11-27T21:34:26.8520815Z Download action repository 'actions/cache@v4' (SHA:0057852bfaa89a56745cba8c7296529d2fc39830)
2025-11-27T21:34:27.0355755Z Download action repository 'astral-sh/setup-uv@v5' (SHA:d4b2f3b6ecc6e67c4457f6d3e41ec42d3d0fcb86)
2025-11-27T21:34:27.8341403Z Complete job name: deploy
2025-11-27T21:34:27.9158069Z ##[group]Run actions/checkout@v4
2025-11-27T21:34:27.9159429Z with:
2025-11-27T21:34:27.9160218Z   repository: adbX/synth-rag
2025-11-27T21:34:27.9161525Z   token: ***
2025-11-27T21:34:27.9162622Z   ssh-strict: true
2025-11-27T21:34:27.9163449Z   ssh-user: git
2025-11-27T21:34:27.9164277Z   persist-credentials: true
2025-11-27T21:34:27.9165216Z   clean: true
2025-11-27T21:34:27.9166038Z   sparse-checkout-cone-mode: true
2025-11-27T21:34:27.9167054Z   fetch-depth: 1
2025-11-27T21:34:27.9167867Z   fetch-tags: false
2025-11-27T21:34:27.9168703Z   show-progress: true
2025-11-27T21:34:27.9169527Z   lfs: false
2025-11-27T21:34:27.9170289Z   submodules: false
2025-11-27T21:34:27.9171129Z   set-safe-directory: true
2025-11-27T21:34:27.9172443Z ##[endgroup]
2025-11-27T21:34:28.0439571Z Syncing repository: adbX/synth-rag
2025-11-27T21:34:28.0443726Z ##[group]Getting Git version info
2025-11-27T21:34:28.0445176Z Working directory is '/home/runner/work/synth-rag/synth-rag'
2025-11-27T21:34:28.0447893Z [command]/usr/bin/git version
2025-11-27T21:34:28.0488234Z git version 2.51.2
2025-11-27T21:34:28.0526221Z ##[endgroup]
2025-11-27T21:34:28.0539099Z Temporarily overriding HOME='/home/runner/work/_temp/199bf180-c1c1-4745-af4a-e9fcbda46a5c' before making global git config changes
2025-11-27T21:34:28.0544709Z Adding repository directory to the temporary git global config as a safe directory
2025-11-27T21:34:28.0559112Z [command]/usr/bin/git config --global --add safe.directory /home/runner/work/synth-rag/synth-rag
2025-11-27T21:34:28.0613222Z Deleting the contents of '/home/runner/work/synth-rag/synth-rag'
2025-11-27T21:34:28.0617993Z ##[group]Initializing the repository
2025-11-27T21:34:28.0622652Z [command]/usr/bin/git init /home/runner/work/synth-rag/synth-rag
2025-11-27T21:34:28.0734308Z hint: Using 'master' as the name for the initial branch. This default branch name
2025-11-27T21:34:28.0739312Z hint: is subject to change. To configure the initial branch name to use in all
2025-11-27T21:34:28.0742604Z hint: of your new repositories, which will suppress this warning, call:
2025-11-27T21:34:28.0745432Z hint:
2025-11-27T21:34:28.0747046Z hint: 	git config --global init.defaultBranch <name>
2025-11-27T21:34:28.0749617Z hint:
2025-11-27T21:34:28.0751459Z hint: Names commonly chosen instead of 'master' are 'main', 'trunk' and
2025-11-27T21:34:28.0755085Z hint: 'development'. The just-created branch can be renamed via this command:
2025-11-27T21:34:28.0757472Z hint:
2025-11-27T21:34:28.0759095Z hint: 	git branch -m <name>
2025-11-27T21:34:28.0762400Z hint:
2025-11-27T21:34:28.0764521Z hint: Disable this message with "git config set advice.defaultBranchName false"
2025-11-27T21:34:28.0767713Z Initialized empty Git repository in /home/runner/work/synth-rag/synth-rag/.git/
2025-11-27T21:34:28.0773771Z [command]/usr/bin/git remote add origin https://github.com/adbX/synth-rag
2025-11-27T21:34:28.0810436Z ##[endgroup]
2025-11-27T21:34:28.0813184Z ##[group]Disabling automatic garbage collection
2025-11-27T21:34:28.0818959Z [command]/usr/bin/git config --local gc.auto 0
2025-11-27T21:34:28.0858407Z ##[endgroup]
2025-11-27T21:34:28.0861866Z ##[group]Setting up auth
2025-11-27T21:34:28.0866749Z [command]/usr/bin/git config --local --name-only --get-regexp core\.sshCommand
2025-11-27T21:34:28.0904646Z [command]/usr/bin/git submodule foreach --recursive sh -c "git config --local --name-only --get-regexp 'core\.sshCommand' && git config --local --unset-all 'core.sshCommand' || :"
2025-11-27T21:34:28.1247353Z [command]/usr/bin/git config --local --name-only --get-regexp http\.https\:\/\/github\.com\/\.extraheader
2025-11-27T21:34:28.1283287Z [command]/usr/bin/git submodule foreach --recursive sh -c "git config --local --name-only --get-regexp 'http\.https\:\/\/github\.com\/\.extraheader' && git config --local --unset-all 'http.https://github.com/.extraheader' || :"
2025-11-27T21:34:28.1536435Z [command]/usr/bin/git config --local --name-only --get-regexp ^includeIf\.gitdir:
2025-11-27T21:34:28.1577803Z [command]/usr/bin/git submodule foreach --recursive git config --local --show-origin --name-only --get-regexp remote.origin.url
2025-11-27T21:34:28.1832828Z [command]/usr/bin/git config --local http.https://github.com/.extraheader AUTHORIZATION: basic ***
2025-11-27T21:34:28.1870968Z ##[endgroup]
2025-11-27T21:34:28.1873792Z ##[group]Fetching the repository
2025-11-27T21:34:28.1882988Z [command]/usr/bin/git -c protocol.version=2 fetch --no-tags --prune --no-recurse-submodules --depth=1 origin +c2a0a527fac93984a2d16f96d81ac1c2b263384d:refs/remotes/origin/main
2025-11-27T21:34:34.4178102Z From https://github.com/adbX/synth-rag
2025-11-27T21:34:34.4179380Z  * [new ref]         c2a0a527fac93984a2d16f96d81ac1c2b263384d -> origin/main
2025-11-27T21:34:34.4214570Z ##[endgroup]
2025-11-27T21:34:34.4215745Z ##[group]Determining the checkout info
2025-11-27T21:34:34.4217060Z ##[endgroup]
2025-11-27T21:34:34.4224372Z [command]/usr/bin/git sparse-checkout disable
2025-11-27T21:34:34.4270547Z [command]/usr/bin/git config --local --unset-all extensions.worktreeConfig
2025-11-27T21:34:34.4298865Z ##[group]Checking out the ref
2025-11-27T21:34:34.4305837Z [command]/usr/bin/git checkout --progress --force -B main refs/remotes/origin/main
2025-11-27T21:34:35.0519958Z Switched to a new branch 'main'
2025-11-27T21:34:35.0523536Z branch 'main' set up to track 'origin/main'.
2025-11-27T21:34:35.0588558Z ##[endgroup]
2025-11-27T21:34:35.0638711Z [command]/usr/bin/git log -1 --format=%H
2025-11-27T21:34:35.0664212Z c2a0a527fac93984a2d16f96d81ac1c2b263384d
2025-11-27T21:34:35.0865340Z ##[group]Run git config user.name github-actions[bot]
2025-11-27T21:34:35.0865920Z [36;1mgit config user.name github-actions[bot][0m
2025-11-27T21:34:35.0866523Z [36;1mgit config user.email 41898282+github-actions[bot]@users.noreply.github.com[0m
2025-11-27T21:34:35.0901702Z shell: /usr/bin/bash -e {0}
2025-11-27T21:34:35.0902453Z ##[endgroup]
2025-11-27T21:34:35.1113414Z ##[group]Run actions/setup-python@v5
2025-11-27T21:34:35.1113934Z with:
2025-11-27T21:34:35.1114240Z   check-latest: false
2025-11-27T21:34:35.1114940Z   token: ***
2025-11-27T21:34:35.1115244Z   update-environment: true
2025-11-27T21:34:35.1115597Z   allow-prereleases: false
2025-11-27T21:34:35.1115863Z   freethreaded: false
2025-11-27T21:34:35.1116186Z ##[endgroup]
2025-11-27T21:34:35.2852405Z [warning]Neither 'python-version' nor 'python-version-file' inputs were supplied. Attempting to find '.python-version' file.
2025-11-27T21:34:35.2855589Z Resolved .python-version as 3.13
2025-11-27T21:34:35.2856535Z ##[group]Installed versions
2025-11-27T21:34:35.2974413Z Successfully set up CPython (3.13.9)
2025-11-27T21:34:35.2981440Z ##[endgroup]
2025-11-27T21:34:35.3086145Z ##[group]Run echo "cache_id=$(date --utc '+%V')" >> $GITHUB_ENV
2025-11-27T21:34:35.3086739Z [36;1mecho "cache_id=$(date --utc '+%V')" >> $GITHUB_ENV[0m
2025-11-27T21:34:35.3356334Z shell: /usr/bin/bash -e {0}
2025-11-27T21:34:35.3356689Z env:
2025-11-27T21:34:35.3357018Z   pythonLocation: /opt/hostedtoolcache/Python/3.13.9/x64
2025-11-27T21:34:35.3418780Z   PKG_CONFIG_PATH: /opt/hostedtoolcache/Python/3.13.9/x64/lib/pkgconfig
2025-11-27T21:34:35.3419541Z   Python_ROOT_DIR: /opt/hostedtoolcache/Python/3.13.9/x64
2025-11-27T21:34:35.3420104Z   Python2_ROOT_DIR: /opt/hostedtoolcache/Python/3.13.9/x64
2025-11-27T21:34:35.3420691Z   Python3_ROOT_DIR: /opt/hostedtoolcache/Python/3.13.9/x64
2025-11-27T21:34:35.3421417Z   LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.13.9/x64/lib
2025-11-27T21:34:35.3421974Z ##[endgroup]
2025-11-27T21:34:35.3614611Z ##[group]Run actions/cache@v4
2025-11-27T21:34:35.3614875Z with:
2025-11-27T21:34:35.3615052Z   key: mkdocs-material-48
2025-11-27T21:34:35.3615263Z   path: ~/.cache
2025-11-27T21:34:35.3615454Z   restore-keys: mkdocs-material-

2025-11-27T21:34:35.3615700Z   enableCrossOsArchive: false
2025-11-27T21:34:35.3615920Z   fail-on-cache-miss: false
2025-11-27T21:34:35.3616127Z   lookup-only: false
2025-11-27T21:34:35.3616319Z   save-always: false
2025-11-27T21:34:35.3616491Z env:
2025-11-27T21:34:35.3616715Z   pythonLocation: /opt/hostedtoolcache/Python/3.13.9/x64
2025-11-27T21:34:35.3617118Z   PKG_CONFIG_PATH: /opt/hostedtoolcache/Python/3.13.9/x64/lib/pkgconfig
2025-11-27T21:34:35.3617501Z   Python_ROOT_DIR: /opt/hostedtoolcache/Python/3.13.9/x64
2025-11-27T21:34:35.3617837Z   Python2_ROOT_DIR: /opt/hostedtoolcache/Python/3.13.9/x64
2025-11-27T21:34:35.3618178Z   Python3_ROOT_DIR: /opt/hostedtoolcache/Python/3.13.9/x64
2025-11-27T21:34:35.3618601Z   LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.13.9/x64/lib
2025-11-27T21:34:35.3618882Z   cache_id: 48
2025-11-27T21:34:35.3619054Z ##[endgroup]
2025-11-27T21:34:35.8606786Z Cache hit for: mkdocs-material-48
2025-11-27T21:34:37.1739994Z Received 4849746 of 25821266 (18.8%), 4.6 MBs/sec
2025-11-27T21:34:37.5457931Z Received 25821266 of 25821266 (100.0%), 17.9 MBs/sec
2025-11-27T21:34:37.5459827Z Cache Size: ~25 MB (25821266 B)
2025-11-27T21:34:37.5493768Z [command]/usr/bin/tar -xf /home/runner/work/_temp/e64538d9-08f3-4b3d-a9b8-824d8be549fb/cache.tzst -P -C /home/runner/work/synth-rag/synth-rag --use-compress-program unzstd
2025-11-27T21:34:37.6224619Z Cache restored successfully
2025-11-27T21:34:37.6355046Z Cache restored from key: mkdocs-material-48
2025-11-27T21:34:37.6499271Z ##[group]Run astral-sh/setup-uv@v5
2025-11-27T21:34:37.6499536Z with:
2025-11-27T21:34:37.6499851Z   github-token: ***
2025-11-27T21:34:37.6500051Z   enable-cache: auto
2025-11-27T21:34:37.6500309Z   cache-dependency-glob: **/uv.lock
**/requirements*.txt

2025-11-27T21:34:37.6500603Z   prune-cache: true
2025-11-27T21:34:37.6500804Z   ignore-nothing-to-cache: false
2025-11-27T21:34:37.6501044Z   ignore-empty-workdir: false
2025-11-27T21:34:37.6501254Z env:
2025-11-27T21:34:37.6501484Z   pythonLocation: /opt/hostedtoolcache/Python/3.13.9/x64
2025-11-27T21:34:37.6501875Z   PKG_CONFIG_PATH: /opt/hostedtoolcache/Python/3.13.9/x64/lib/pkgconfig
2025-11-27T21:34:37.6502509Z   Python_ROOT_DIR: /opt/hostedtoolcache/Python/3.13.9/x64
2025-11-27T21:34:37.6502902Z   Python2_ROOT_DIR: /opt/hostedtoolcache/Python/3.13.9/x64
2025-11-27T21:34:37.6503445Z   Python3_ROOT_DIR: /opt/hostedtoolcache/Python/3.13.9/x64
2025-11-27T21:34:37.6503813Z   LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.13.9/x64/lib
2025-11-27T21:34:37.6504114Z   cache_id: 48
2025-11-27T21:34:37.6504289Z ##[endgroup]
2025-11-27T21:34:38.2957093Z Downloading uv from "https://github.com/astral-sh/uv/releases/download/0.9.13/uv-x86_64-unknown-linux-gnu.tar.gz" ...
2025-11-27T21:34:38.7435399Z [command]/usr/bin/tar xz --warning=no-unknown-keyword --overwrite -C /home/runner/work/_temp/c9ff3835-7444-4e24-b372-e4a21c5e6b1f -f /home/runner/work/_temp/c46f7234-2a3b-4ab3-beac-2fe6918e0da0
2025-11-27T21:34:39.2073903Z Added /home/runner/.local/bin to the path
2025-11-27T21:34:39.2074560Z Added /opt/hostedtoolcache/uv/0.9.13/x86_64 to the path
2025-11-27T21:34:39.2084018Z Set UV_CACHE_DIR to /home/runner/work/_temp/setup-uv-cache
2025-11-27T21:34:39.2084628Z Successfully installed uv version 0.9.13
2025-11-27T21:34:39.2085406Z Searching files using cache dependency glob: **/uv.lock,**/requirements*.txt
2025-11-27T21:34:39.2367621Z /home/runner/work/synth-rag/synth-rag/uv.lock
2025-11-27T21:34:39.2398844Z Found 1 files to hash.
2025-11-27T21:34:40.1289581Z Trying to restore uv cache from GitHub Actions cache with key: setup-uv-1-x86_64-unknown-linux-gnu-3.13.9-f9d70c9524f4ed9e43fad8ecdf1849eb22ec81958a1ae85a97439735b1d1f84c
2025-11-27T21:34:40.3677207Z No GitHub Actions cache found for key: setup-uv-1-x86_64-unknown-linux-gnu-3.13.9-f9d70c9524f4ed9e43fad8ecdf1849eb22ec81958a1ae85a97439735b1d1f84c
2025-11-27T21:34:40.3800407Z ##[group]Run uv sync --only-group docs
2025-11-27T21:34:40.3800947Z [36;1muv sync --only-group docs[0m
2025-11-27T21:34:40.3849323Z shell: /usr/bin/bash -e {0}
2025-11-27T21:34:40.3849705Z env:
2025-11-27T21:34:40.3850124Z   pythonLocation: /opt/hostedtoolcache/Python/3.13.9/x64
2025-11-27T21:34:40.3850840Z   PKG_CONFIG_PATH: /opt/hostedtoolcache/Python/3.13.9/x64/lib/pkgconfig
2025-11-27T21:34:40.3851540Z   Python_ROOT_DIR: /opt/hostedtoolcache/Python/3.13.9/x64
2025-11-27T21:34:40.3852175Z   Python2_ROOT_DIR: /opt/hostedtoolcache/Python/3.13.9/x64
2025-11-27T21:34:40.3853083Z   Python3_ROOT_DIR: /opt/hostedtoolcache/Python/3.13.9/x64
2025-11-27T21:34:40.3853706Z   LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.13.9/x64/lib
2025-11-27T21:34:40.3854218Z   cache_id: 48
2025-11-27T21:34:40.3854609Z   UV_CACHE_DIR: /home/runner/work/_temp/setup-uv-cache
2025-11-27T21:34:40.3855095Z ##[endgroup]
2025-11-27T21:34:40.4228231Z Using CPython 3.13.9 interpreter at: /opt/hostedtoolcache/Python/3.13.9/x64/bin/python3.13
2025-11-27T21:34:40.4233101Z Creating virtual environment at: .venv
2025-11-27T21:34:40.4345930Z Resolved 251 packages in 0.63ms
2025-11-27T21:34:40.4564909Z Downloading babel (9.7MiB)
2025-11-27T21:34:40.4568963Z Downloading pygments (1.2MiB)
2025-11-27T21:34:40.4576404Z Downloading mkdocs (3.7MiB)
2025-11-27T21:34:40.4642155Z Downloading mkdocs-material (8.9MiB)
2025-11-27T21:34:40.6754219Z  Downloaded mkdocs
2025-11-27T21:34:40.7099913Z  Downloaded pygments
2025-11-27T21:34:40.9050112Z  Downloaded babel
2025-11-27T21:34:42.5526514Z  Downloaded mkdocs-material
2025-11-27T21:34:42.5529588Z Prepared 33 packages in 2.11s
2025-11-27T21:34:42.9148680Z Installed 33 packages in 361ms
2025-11-27T21:34:42.9153766Z  + babel==2.17.0
2025-11-27T21:34:42.9154587Z  + backrefs==6.1
2025-11-27T21:34:42.9155618Z  + certifi==2025.11.12
2025-11-27T21:34:42.9156606Z  + charset-normalizer==3.4.4
2025-11-27T21:34:42.9157726Z  + click==8.3.1
2025-11-27T21:34:42.9158623Z  + colorama==0.4.6
2025-11-27T21:34:42.9159558Z  + ghp-import==2.1.0
2025-11-27T21:34:42.9162502Z  + griffe==1.15.0
2025-11-27T21:34:42.9167369Z  + idna==3.11
2025-11-27T21:34:42.9168386Z  + jinja2==3.1.6
2025-11-27T21:34:42.9169260Z  + markdown==3.10
2025-11-27T21:34:42.9170194Z  + markupsafe==3.0.3
2025-11-27T21:34:42.9171945Z  + mergedeep==1.3.4
2025-11-27T21:34:42.9172724Z  + mkdocs==1.6.1
2025-11-27T21:34:42.9173837Z  + mkdocs-autorefs==1.4.3
2025-11-27T21:34:42.9174212Z  + mkdocs-get-deps==0.2.0
2025-11-27T21:34:42.9174846Z  + mkdocs-material==9.7.0
2025-11-27T21:34:42.9175232Z  + mkdocs-material-extensions==1.3.1
2025-11-27T21:34:42.9175652Z  + mkdocstrings==1.0.0
2025-11-27T21:34:42.9175997Z  + mkdocstrings-python==2.0.0
2025-11-27T21:34:42.9176374Z  + packaging==25.0
2025-11-27T21:34:42.9176683Z  + paginate==0.5.7
2025-11-27T21:34:42.9176976Z  + pathspec==0.12.1
2025-11-27T21:34:42.9177323Z  + platformdirs==4.5.0
2025-11-27T21:34:42.9177652Z  + pygments==2.19.2
2025-11-27T21:34:42.9177967Z  + pymdown-extensions==10.17.2
2025-11-27T21:34:42.9178366Z  + python-dateutil==2.9.0.post0
2025-11-27T21:34:42.9178735Z  + pyyaml==6.0.3
2025-11-27T21:34:42.9179048Z  + pyyaml-env-tag==1.1
2025-11-27T21:34:42.9179358Z  + requests==2.32.5
2025-11-27T21:34:42.9179646Z  + six==1.17.0
2025-11-27T21:34:42.9179925Z  + urllib3==2.5.0
2025-11-27T21:34:42.9180210Z  + watchdog==6.0.0
2025-11-27T21:34:42.9223384Z ##[group]Run uv run mkdocs gh-deploy --force
2025-11-27T21:34:42.9223754Z [36;1muv run mkdocs gh-deploy --force[0m
2025-11-27T21:34:42.9256791Z shell: /usr/bin/bash -e {0}
2025-11-27T21:34:42.9257021Z env:
2025-11-27T21:34:42.9257272Z   pythonLocation: /opt/hostedtoolcache/Python/3.13.9/x64
2025-11-27T21:34:42.9257698Z   PKG_CONFIG_PATH: /opt/hostedtoolcache/Python/3.13.9/x64/lib/pkgconfig
2025-11-27T21:34:42.9258089Z   Python_ROOT_DIR: /opt/hostedtoolcache/Python/3.13.9/x64
2025-11-27T21:34:42.9258426Z   Python2_ROOT_DIR: /opt/hostedtoolcache/Python/3.13.9/x64
2025-11-27T21:34:42.9258766Z   Python3_ROOT_DIR: /opt/hostedtoolcache/Python/3.13.9/x64
2025-11-27T21:34:42.9259114Z   LD_LIBRARY_PATH: /opt/hostedtoolcache/Python/3.13.9/x64/lib
2025-11-27T21:34:42.9259395Z   cache_id: 48
2025-11-27T21:34:42.9259617Z   UV_CACHE_DIR: /home/runner/work/_temp/setup-uv-cache
2025-11-27T21:34:42.9259901Z ##[endgroup]
2025-11-27T21:34:43.0581718Z    Building synth-rag @ file:///home/runner/work/synth-rag/synth-rag
2025-11-27T21:34:43.0710430Z Downloading nvidia-cufft-cu12 (184.2MiB)
2025-11-27T21:34:43.0733568Z Downloading jupyterlab (11.8MiB)
2025-11-27T21:34:43.0734393Z Downloading sympy (6.0MiB)
2025-11-27T21:34:43.0735913Z Downloading sqlalchemy (3.1MiB)
2025-11-27T21:34:43.0736335Z Downloading tokenizers (3.1MiB)
2025-11-27T21:34:43.0738626Z Downloading pydantic-core (2.0MiB)
2025-11-27T21:34:43.0743254Z Downloading networkx (1.9MiB)
2025-11-27T21:34:43.0758370Z Downloading jedi (1.5MiB)
2025-11-27T21:34:43.0759060Z Downloading setuptools (1.1MiB)
2025-11-27T21:34:43.0864795Z Downloading debugpy (4.1MiB)
2025-11-27T21:34:43.0868806Z Downloading widgetsnbextension (2.1MiB)
2025-11-27T21:34:43.0873134Z Downloading nvidia-cufile-cu12 (1.1MiB)
2025-11-27T21:34:43.0878693Z Downloading tiktoken (1.1MiB)
2025-11-27T21:34:43.0885488Z Downloading gradio (20.6MiB)
2025-11-27T21:34:43.0889636Z Downloading triton (148.4MiB)
2025-11-27T21:34:43.0894619Z Downloading pandas (11.7MiB)
2025-11-27T21:34:43.0898796Z Downloading nvidia-cublas-cu12 (566.8MiB)
2025-11-27T21:34:43.0903570Z Downloading brotli (1.4MiB)
2025-11-27T21:34:43.0906590Z Downloading notebook (13.8MiB)
2025-11-27T21:34:43.0910098Z Downloading zstandard (5.2MiB)
2025-11-27T21:34:43.0913443Z Downloading aiohttp (1.7MiB)
2025-11-27T21:34:43.0916745Z Downloading grpcio-tools (2.5MiB)
2025-11-27T21:34:43.0919984Z Downloading nvidia-cusparse-cu12 (274.9MiB)
2025-11-27T21:34:43.0923481Z Downloading scipy (34.0MiB)
2025-11-27T21:34:43.0926859Z Downloading transformers (11.4MiB)
2025-11-27T21:34:43.0930168Z Downloading pypdfium2 (2.9MiB)
2025-11-27T21:34:43.0933732Z Downloading langchain-community (2.4MiB)
2025-11-27T21:34:43.0938638Z Downloading nvidia-curand-cu12 (60.7MiB)
2025-11-27T21:34:43.0941971Z Downloading onnx (17.4MiB)
2025-11-27T21:34:43.0946690Z Downloading torch (846.8MiB)
2025-11-27T21:34:43.0950422Z Downloading hf-xet (3.2MiB)
2025-11-27T21:34:43.0954250Z Downloading semantic-text-splitter (8.0MiB)
2025-11-27T21:34:43.0957695Z Downloading pillow (4.3MiB)
2025-11-27T21:34:43.0961264Z Downloading nvidia-cusolver-cu12 (255.1MiB)
2025-11-27T21:34:43.0965857Z Downloading nvidia-cuda-cupti-cu12 (9.8MiB)
2025-11-27T21:34:43.0970168Z Downloading nvidia-nvjitlink-cu12 (37.4MiB)
2025-11-27T21:34:43.0974221Z Downloading nvidia-cuda-nvrtc-cu12 (84.0MiB)
2025-11-27T21:34:43.0978085Z Downloading nvidia-nccl-cu12 (307.4MiB)
2025-11-27T21:34:43.0982116Z Downloading nvidia-cusparselt-cu12 (273.9MiB)
2025-11-27T21:34:43.0986290Z Downloading nvidia-cudnn-cu12 (674.0MiB)
2025-11-27T21:34:43.0990421Z Downloading torchvision (8.2MiB)
2025-11-27T21:34:43.0994284Z Downloading onnxruntime (16.6MiB)
2025-11-27T21:34:43.0998068Z Downloading scikit-learn (9.0MiB)
2025-11-27T21:34:43.1003364Z Downloading pymupdf (23.0MiB)
2025-11-27T21:34:43.1007426Z Downloading grpcio (6.3MiB)
2025-11-27T21:34:43.8996966Z  Downloaded nvidia-cufile-cu12
2025-11-27T21:34:43.9393691Z  Downloaded tiktoken
2025-11-27T21:34:44.0713875Z  Downloaded brotli
2025-11-27T21:34:44.2876034Z  Downloaded aiohttp
2025-11-27T21:34:44.8363737Z  Downloaded pydantic-core
2025-11-27T21:34:44.8487800Z  Downloaded setuptools
2025-11-27T21:34:44.9294528Z  Downloaded widgetsnbextension
2025-11-27T21:34:45.1185183Z  Downloaded networkx
2025-11-27T21:34:45.4864430Z  Downloaded grpcio-tools
2025-11-27T21:34:45.7139101Z  Downloaded pypdfium2
2025-11-27T21:34:45.8339009Z  Downloaded sqlalchemy
2025-11-27T21:34:45.8576946Z  Downloaded hf-xet
2025-11-27T21:34:45.8823094Z  Downloaded tokenizers
2025-11-27T21:34:46.9056043Z  Downloaded debugpy
2025-11-27T21:34:47.1112588Z  Downloaded pillow
2025-11-27T21:34:47.1163768Z    Building mmh3==4.1.0
2025-11-27T21:34:47.5683406Z  Downloaded langchain-community
2025-11-27T21:34:47.6857554Z  Downloaded zstandard
2025-11-27T21:34:47.6885806Z    Building ml-dtypes==0.4.1
2025-11-27T21:34:48.4556694Z       Built synth-rag @ file:///home/runner/work/synth-rag/synth-rag
2025-11-27T21:34:48.4761281Z  Downloaded grpcio
2025-11-27T21:34:48.7006699Z  Downloaded sympy
2025-11-27T21:34:49.3745598Z  Downloaded semantic-text-splitter
2025-11-27T21:34:49.4598473Z  Downloaded torchvision
2025-11-27T21:34:49.4869109Z  Downloaded jedi
2025-11-27T21:34:50.2140060Z  Downloaded scikit-learn
2025-11-27T21:34:50.4351339Z  Downloaded nvidia-cuda-cupti-cu12
2025-11-27T21:34:51.3884672Z  Downloaded transformers
2025-11-27T21:34:51.6573713Z  Downloaded jupyterlab
2025-11-27T21:34:52.6508378Z  Downloaded pandas
2025-11-27T21:34:53.1033336Z       Built mmh3==4.1.0
2025-11-27T21:34:53.1539384Z  Downloaded notebook
2025-11-27T21:34:53.9618095Z  Downloaded onnxruntime
2025-11-27T21:34:55.0040630Z  Downloaded gradio
2025-11-27T21:34:56.4738474Z  Downloaded pymupdf
2025-11-27T21:34:59.3258174Z  Downloaded nvidia-nvjitlink-cu12
2025-11-27T21:34:59.9115123Z  Downloaded scipy
2025-11-27T21:35:01.7703129Z  Downloaded nvidia-curand-cu12
2025-11-27T21:35:04.3065543Z  Downloaded nvidia-cuda-nvrtc-cu12
2025-11-27T21:35:07.0619712Z  Downloaded onnx
2025-11-27T21:35:12.7823531Z  Downloaded triton
2025-11-27T21:35:13.1898305Z    Building numpy==1.26.4
2025-11-27T21:35:14.5346253Z  Downloaded nvidia-cufft-cu12
2025-11-27T21:35:22.0029646Z  Downloaded nvidia-cusolver-cu12
2025-11-27T21:35:24.0325709Z  Downloaded nvidia-cusparselt-cu12
2025-11-27T21:35:24.0746930Z  Downloaded nvidia-cusparse-cu12
2025-11-27T21:35:26.8824766Z  Downloaded nvidia-nccl-cu12
2025-11-27T21:35:38.6773768Z  Downloaded nvidia-cublas-cu12
2025-11-27T21:35:41.7661839Z  Downloaded nvidia-cudnn-cu12
2025-11-27T21:35:44.6105832Z  Downloaded torch
2025-11-27T21:35:49.2957764Z       Built ml-dtypes==0.4.1
2025-11-27T21:36:25.0995747Z ##[error]The operation was canceled.
2025-11-27T21:36:25.1220641Z Post job cleanup.
2025-11-27T21:36:25.4923220Z [command]/usr/bin/git version
2025-11-27T21:36:25.5023010Z git version 2.51.2
2025-11-27T21:36:25.5134806Z Temporarily overriding HOME='/home/runner/work/_temp/fdd3fe43-d97c-4358-b987-0a6e253bb501' before making global git config changes
2025-11-27T21:36:26.6198794Z Adding repository directory to the temporary git global config as a safe directory
2025-11-27T21:36:26.6380934Z [command]/usr/bin/git config --global --add safe.directory /home/runner/work/synth-rag/synth-rag
2025-11-27T21:36:26.6383633Z [command]/usr/bin/git config --local --name-only --get-regexp core\.sshCommand
2025-11-27T21:36:26.6414337Z [command]/usr/bin/git submodule foreach --recursive sh -c "git config --local --name-only --get-regexp 'core\.sshCommand' && git config --local --unset-all 'core.sshCommand' || :"
2025-11-27T21:36:26.6416839Z [command]/usr/bin/git config --local --name-only --get-regexp http\.https\:\/\/github\.com\/\.extraheader
2025-11-27T21:36:26.6417843Z http.https://github.com/.extraheader
2025-11-27T21:36:26.6419446Z [command]/usr/bin/git config --local --unset-all http.https://github.com/.extraheader
2025-11-27T21:36:26.6421817Z [command]/usr/bin/git submodule foreach --recursive sh -c "git config --local --name-only --get-regexp 'http\.https\:\/\/github\.com\/\.extraheader' && git config --local --unset-all 'http.https://github.com/.extraheader' || :"
2025-11-27T21:36:26.6424419Z [command]/usr/bin/git config --local --name-only --get-regexp ^includeIf\.gitdir:
2025-11-27T21:36:26.6434271Z [command]/usr/bin/git submodule foreach --recursive git config --local --show-origin --name-only --get-regexp remote.origin.url
2025-11-27T21:36:26.6550190Z Cleaning up orphan processes
2025-11-27T21:36:26.7318874Z Terminate orphan process: pid (2075) (uv)
2025-11-27T21:36:26.7439890Z Terminate orphan process: pid (2178) (python)
2025-11-27T21:36:26.7544465Z Terminate orphan process: pid (3327) (ninja)
2025-11-27T21:36:26.7581884Z Terminate orphan process: pid (3810) (sh)
2025-11-27T21:36:26.7676288Z Terminate orphan process: pid (3811) (cc)
2025-11-27T21:36:26.7750394Z Terminate orphan process: pid (3812) (cc1)
2025-11-27T21:36:26.7852961Z Terminate orphan process: pid (3827) (sh)
2025-11-27T21:36:26.7921862Z Terminate orphan process: pid (3828) (cc)
2025-11-27T21:36:26.8010941Z Terminate orphan process: pid (3829) (cc1)
2025-11-27T21:36:26.8060589Z Terminate orphan process: pid (3919) (sh)
2025-11-27T21:36:26.8092004Z Terminate orphan process: pid (3920) (cc)
```