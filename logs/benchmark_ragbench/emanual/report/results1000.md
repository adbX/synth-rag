```
ed 3. Proceeding with 1 generations.
Evaluating: 100%|██████████████████████████████████████████████████| 2000/2000 [53:33<00:00,  1.61s/it]
✓ RAGAS metrics computed
  - Mean faithfulness: 0.832
  - Mean answer_relevancy: 0.603

================================================================================
Computing TruLens Metrics
================================================================================
Evaluating 1000 examples with TruLens...
Note: Processing sequentially (non-concurrent)
TruLens evaluation:   2%|▉                                         | 23/1000 [03:15<2:38:31,  9.74s/it]/Users/adb/stuff/gitclones/synth-rag/.venv/lib/python3.13/site-packages/trulens/feedback/llm_provider.py:2141: UserWarning: Failed to process and remove trivial statements. Proceeding with all statements.
  warnings.warn(
TruLens evaluation: 100%|████████████████████████████████████████| 1000/1000 [2:40:11<00:00,  9.61s/it]

✓ TruLens metrics computed
  - Mean groundedness: 0.792
  - Mean context_relevance: 0.662

================================================================================
Computing Aggregate Metrics
================================================================================
✓ Processed 1000 examples
✓ Mean query time: 0.064s
✓ Mean generation time: 3.185s
✓ Mean total time: 3.248s
✓ RAGAS faithfulness: 0.832 ± 0.245
✓ RAGAS context relevancy: 0.603 ± 0.455
✓ TruLens groundedness: 0.792 ± 0.137
✓ TruLens context relevance: 0.662 ± 0.350

✓ Saved detailed results to: /Users/adb/stuff/gitclones/synth-rag/logs/benchmark_ragbench/emanual/20251206_130240_detailed_results.jsonl
✓ Saved aggregate metrics to: /Users/adb/stuff/gitclones/synth-rag/logs/benchmark_ragbench/emanual/20251206_130240_metrics.json
✓ Saved summary CSV to: /Users/adb/stuff/gitclones/synth-rag/logs/benchmark_ragbench/emanual/20251206_130240_summary.csv

================================================================================
✅ Metrics computation complete!
================================================================================
adb ~/stuff/gitclones/synth-rag main ≡*?9  3h 33m 53.019s 
```