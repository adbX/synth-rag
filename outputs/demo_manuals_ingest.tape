# VHS Tape: MIDI Manuals Ingestion Demo
# Demonstrates ingesting PDF manuals into Qdrant using ColPali + hybrid embeddings

Output outputs/demo_manuals_ingest.gif
Output outputs/demo_manuals_ingest.mp4

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
Type "# Ingest PDF manuals into Qdrant with ColPali embeddings"
Enter
Sleep 500ms

Type "uv run python -m synth_rag.manuals_ingest --subset test --collection midi_manuals --device mps --recreate-collection --clear-tmp"
Enter
Sleep 1s

# Wait for the ingestion to complete (10 min timeout for long-running process)
Wait+Screen@10m /Ingestion complete

# Give time to see the final output
Sleep 3s

