# VHS Tape: RAGBench Dataset Ingestion Demo
# Demonstrates ingesting RAGBench emanual dataset into Qdrant for benchmarking

Output outputs/demo_benchmark_ingest.gif
Output outputs/demo_benchmark_ingest.mp4

Require uv

# Terminal settings
Set Shell "zsh"
Set FontSize 18
Set FontFamily "JetBrains Mono"
Set Width 1400
Set Height 700
Set Theme "Catppuccin Mocha"
Set WindowBar Colorful
Set Padding 20
Set TypingSpeed 40ms
Set PlaybackSpeed 3.0

# Start the demo
Sleep 500ms

# Show the command being typed
Type "# Ingest RAGBench emanual dataset for benchmarking"
Enter
Sleep 500ms

Type "uv run python -m synth_rag.benchmark_ingest \"
Enter
Type "    --dataset emanual \"
Enter
Type "    --split all \"
Enter
Type "    --recreate-collection"
Sleep 1s
Enter

# Wait for the ingestion to complete (15 min timeout for large dataset)
Wait+Screen@15m /Ingestion complete|Total points|points uploaded|examples processed/

# Give time to see the final output
Sleep 3s

