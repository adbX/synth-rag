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