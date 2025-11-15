<!-- 67e3db95-45c3-48be-8a24-b89a215edfc6 f9684a02-cd06-445f-8173-dbb677a0aef0 -->
# Integrate VLM Pipeline with GraniteDocling MLX

## Changes Required

### 1. Update Cell 0 Imports

Add the necessary imports for VLM pipeline in `/Users/adb/stuff/gitclones/qdrant-init/src/qdrant_init/docling_qdrant.ipynb`:

- Import `vlm_model_specs` from `docling.datamodel`
- Import `VlmPipelineOptions` from `docling.datamodel.pipeline_options`  
- Import `PdfFormatOption` from `docling.document_converter`
- Import `VlmPipeline` from `docling.pipeline.vlm_pipeline`

### 2. Update Cell 1 DocumentConverter Configuration

Replace the simple `DocumentConverter(allowed_formats=[InputFormat.PDF])` initialization with:

```python
pipeline_options = VlmPipelineOptions(
    vlm_options=vlm_model_specs.GRANITEDOCLING_MLX,
)

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_cls=VlmPipeline,
            pipeline_options=pipeline_options,
        ),
    }
)
```

This configures the converter to use the VLM pipeline with the granite-docling-258M-mlx model and macOS MPS acceleration.

### 3. Dependencies Note

The `pyproject.toml` already includes `docling[vlm]>=2.46.0`. For MLX support on macOS, ensure the MLX backend is available. If the first run shows missing dependencies, may need to add MLX-specific packages.

## Files to Modify

- `/Users/adb/stuff/gitclones/qdrant-init/src/qdrant_init/docling_qdrant.ipynb` (Cells 0 and 1)

## Expected Outcome

The notebook will use the vision-language model to convert PDFs, which should provide better handling of complex layouts, tables, and figures in your MIDI synthesizer manuals.