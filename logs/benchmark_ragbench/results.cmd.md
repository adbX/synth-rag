```
➜ uv run python -m synth_rag.benchmark_metrics --results-file /Users/adb/stuff/gitclones/synth-rag/logs/benchmark_ragbench/emanual/20251206_104821_raw_results.jsonl
/Users/adb/stuff/gitclones/synth-rag/.venv/lib/python3.13/site-packages/munch/__init__.py:24: UserWarning: pkg_resources is deprecated as an API. See https://setuptools.pypa.io/en/latest/pkg_resources.html. The pkg_resources package is slated for removal as early as 2025-11-30. Refrain from using this package or pin to Setuptools<81.
  import pkg_resources
Loading results from: /Users/adb/stuff/gitclones/synth-rag/logs/benchmark_ragbench/emanual/20251206_104821_raw_results.jsonl
✓ Loaded 20 results

================================================================================
Computing RAGAS Metrics
================================================================================
Evaluating 20 examples with RAGAS...
Evaluating:  35%|███████████████████████                                           | 14/40 [01:00<01:25,  3.28s/it]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating:  55%|████████████████████████████████████▎                             | 22/40 [01:06<00:31,  1.74s/it]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating:  57%|█████████████████████████████████████▉                            | 23/40 [01:08<00:30,  1.80s/it]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating:  60%|███████████████████████████████████████▌                          | 24/40 [01:12<00:33,  2.09s/it]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating:  65%|██████████████████████████████████████████▉                       | 26/40 [01:16<00:30,  2.14s/it]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating:  68%|████████████████████████████████████████████▌                     | 27/40 [01:19<00:28,  2.21s/it]LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
LLM returned 1 generations instead of requested 3. Proceeding with 1 generations.
Evaluating: 100%|██████████████████████████████████████████████████████████████████| 40/40 [01:43<00:00,  2.58s/it]
✓ RAGAS metrics computed
  - Mean faithfulness: 0.732
  - Mean answer_relevancy: 0.799

================================================================================
Computing TruLens Metrics
================================================================================
Evaluating 20 examples with TruLens...
Note: Processing sequentially (non-concurrent)
TruLens evaluation:   0%|                                                                   | 0/20 [00:00<?, ?it/s]/Users/adb/stuff/gitclones/synth-rag/.venv/lib/python3.13/site-packages/trulens/feedback/llm_provider.py:2141: UserWarning: Failed to process and remove trivial statements. Proceeding with all statements.
  warnings.warn(
TruLens evaluation: 100%|██████████████████████████████████████████████████████████| 20/20 [03:20<00:00, 10.03s/it]

✓ TruLens metrics computed
  - Mean groundedness: 0.756
  - Mean context_relevance: 0.700

================================================================================
Computing Aggregate Metrics
================================================================================
✓ Processed 20 examples
✓ Mean query time: 0.061s
✓ Mean generation time: 3.602s
✓ Mean total time: 3.662s
✓ RAGAS faithfulness: 0.732 ± 0.314
✓ RAGAS context relevancy: 0.799 ± 0.337
✓ TruLens groundedness: 0.756 ± 0.130
✓ TruLens context relevance: 0.700 ± 0.314
✓ Hallucination AUROC (RAGAS): 0.944
✓ Relevance RMSE (RAGAS): 0.687

✓ Saved detailed results to: /Users/adb/stuff/gitclones/synth-rag/logs/benchmark_ragbench/emanual/20251206_104821_detailed_results.jsonl
✓ Saved aggregate metrics to: /Users/adb/stuff/gitclones/synth-rag/logs/benchmark_ragbench/emanual/20251206_104821_metrics.json
✓ Saved summary CSV to: /Users/adb/stuff/gitclones/synth-rag/logs/benchmark_ragbench/emanual/20251206_104821_summary.csv

================================================================================
✅ Metrics computation complete!
================================================================================
```