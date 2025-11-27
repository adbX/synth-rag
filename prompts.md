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

I have mkdocs with uv `uv run mkdocs serve`. Create a comprehensive set of documentation using my project readme `README.md` and the scripts in the `src/synth_rag/` folder.
