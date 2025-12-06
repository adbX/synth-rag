# VHS Tape: Benchmark Runner Demo
# Demonstrates running RAGBench evaluation with hybrid search and LLM responses

Output outputs/demo_benchmark_runner.gif
Output outputs/demo_benchmark_runner.mp4

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
Type "# Run RAGBench evaluation on emanual test split"
Enter
Sleep 500ms

Type "uv run python -m synth_rag.benchmark_runner \"
Enter
Type "    --dataset emanual \"
Enter
Type "    --split test \"
Enter
Type "    --model gpt-4o-mini \"
Enter
Type "    --top-k 5"
Sleep 1s
Enter

# Wait for the benchmark to complete (30 min timeout for full evaluation)
Wait+Screen@30m /Benchmark complete|Results saved|examples evaluated|Summary:/

# Give time to see the final metrics output
Sleep 5s

