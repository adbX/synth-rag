Your role is to help benchmark the performance of my `synth-rag` system using the `ragbench` dataset from huggingface (https://huggingface.co/datasets/galileo-ai/ragbench). It's README.md file is available in `prompts/ragbench-README.md`. I have queried 3 random rows from the dataset, from load_dataset("rungalileo/ragbench", "emanual", split="train") in particular, and the complete rows are available in `prompts/ragbench-random3.json`

Your task is understand the usage of `synth-rag` by reading its documentation in `/docs`, and the code in `src/` to extend it to benchmark the performance of the system. I want to benchmark the system's performance on the `emanual` dataset of `ragbench` in particular.

- The code should be simple and minimal, and should be an optional/independent feature to the main functionality of synth-rag
- Create a robust logger/saver to record all relevant data for benchmark queries/parameters/settings, timestamps and results with.
- Make it extensible to also test other sub-datasets of `ragbench` if I want to.
- Create a new qdrant collection for each ragbench sub-dataset and a way to just ingest the collection (prior to benchmarking) as a separate command.
- Measure all the relevant metrics for the benchmark used in the ragbanch dataset & readme
- Test both modes of synth-rag: the agentic and hybrid search methods

I have installed ragas and trulens using uv already. So use those libraries. I have also added the ragbench code for evaluation to `prompts/ragbench-main/ragbench` which you can use to develop the metric evaluation scripts. Actually, just evaluate the hybrid search mode for now, don't evaluate agentic mode yet.

benchmark-eval embed

<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe id="js_video_iframe" src="https://jumpshare.com/embed/GtThOHIS6Csq619rOmp3" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>

benchmark ingest embed

<div style="position: relative; padding-bottom: 56.25%; height: 0;"><iframe id="js_video_iframe" src="https://jumpshare.com/embed/jXbKbukrL5QbcextcHiQ" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>

https://streamable.com/qi8kw0?src=player-page-share