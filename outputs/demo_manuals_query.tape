# VHS Tape: Querying Manuals Demo
# Demonstrates hybrid search over indexed manuals with LLM response generation

Output outputs/demo_manuals_query.gif
Output outputs/demo_manuals_query.mp4

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
Set PlaybackSpeed 2.0

# Start the demo
Sleep 500ms

# Show the command being typed
Type "# Query indexed manuals using hybrid search (ColPali + dense + sparse)"
Enter
Sleep 500ms

Type "uv run python -m synth_rag.manuals_query --question "In the FM Drum Machine of Digitone II, what are all the ways to increase the decay time?" --collection midi_manuals --top-k 5 --device mps"
Enter
Sleep 1s
Enter

# Wait for the response to complete (2 min timeout)
Wait+Screen@2m /Response:|Answer:|retrieved|Results:/

# Give time to read the full response
Sleep 5s

