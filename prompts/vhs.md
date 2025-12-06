My goal is to create a few demo videos to showcase the cli aspects of the `synth-rag` project. Use the `vhs` (https://github.com/charmbracelet/vhs) tool to create the videos. I have provided the readme documentation and examples in the `prompts/vhs-readme.md` file. Use my projects' `docs` folder for reference on how to use the cli commands. I want videos for:

- Ingesting midi manuals: `uv run python -m synth_rag.manuals_ingest --subset test --collection midi_manuals --device mps --recreate-collection --clear-tmp`
- Querying: `uv run python -m synth_rag.manuals_query \
    --question "In the FM Drum Machine of Digitone II, what are all the ways to increase the decay time?" \
    --collection midi_manuals \
    --top-k 5 \
    --device mps`
- Ingesting emanual of RAGBench: `uv run python -m synth_rag.benchmark_ingest \
    --dataset emanual \
    --split all \
    --recreate-collection`
- Running the benchmark on emanual of RAGBench: `uv run python -m synth_rag.benchmark_runner \
    --dataset emanual \
    --split test \
    --model gpt-4o-mini \
    --top-k 5`