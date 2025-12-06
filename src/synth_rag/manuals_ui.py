#!/usr/bin/env python3
"""
Gradio UI for MIDI Manuals RAG Chatbot.
"""
from __future__ import annotations

import os
import gradio as gr
from langchain_core.messages import ToolMessage

from synth_rag.manuals_agent import (
    create_agent_graph,
    ManualsRetriever,
)
from synth_rag.settings import get_api_settings


def chat_function(message: str, history: list, model: str, collection: str, device: str, top_k: int):
    """
    Chat function for Gradio ChatInterface.
    
    Args:
        message: User's question
        history: Chat history in OpenAI format
        model: OpenAI model name
        collection: Qdrant collection name
        device: Device for ColPali model
        top_k: Number of results to retrieve
    """
    # Set OpenAI API key
    api_settings = get_api_settings()
    os.environ["OPENAI_API_KEY"] = api_settings.openai_key
    
    # Create agent graph
    graph = create_agent_graph(
        collection_name=collection,
        device=device,
        top_k=top_k,
        model=model,
    )
    
    # Stream agent responses
    accumulated_response = ""
    tool_messages = []
    
    for event in graph.stream({"messages": [("user", message)]}):
        for value in event.values():
            last_message = value["messages"][-1]
            
            # Handle tool calls
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    tool_name = tool_call.get("name", "unknown")
                    
                    # Show inline status based on tool
                    if "manuals" in tool_name.lower():
                        status = "🔍 Searching PDF manuals..."
                    elif "web" in tool_name.lower() or "search" in tool_name.lower():
                        status = "🌐 Searching the web..."
                    else:
                        status = f"🔧 Using tool: {tool_name}..."
                    
                    if status not in tool_messages:
                        tool_messages.append(status)
                        accumulated_response = "\n".join(tool_messages)
                        yield accumulated_response
            
            # Handle tool responses (optional: can show brief feedback)
            elif isinstance(last_message, ToolMessage):
                # Optionally show that tool completed
                pass
            
            # Handle final answer
            elif hasattr(last_message, "content") and last_message.content:
                if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
                    # This is the final answer
                    yield last_message.content
                    return


def create_ui():
    """Create the Gradio UI."""
    
    with gr.Blocks(title="MIDI Manuals RAG Chatbot") as demo:
        gr.Markdown("# 🎹 MIDI Manuals RAG Chatbot")
        gr.Markdown(
            "Ask questions about MIDI synthesizer manuals. "
            "The agent can search PDF manuals or the web to answer your questions."
        )
        
        # Configuration row
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"],
                value="gpt-4o-mini",
                label="Model",
                info="OpenAI model to use",
            )
            collection_textbox = gr.Textbox(
                value="midi_manuals",
                label="Collection",
                info="Qdrant collection name",
            )
            device_dropdown = gr.Dropdown(
                choices=["mps", "cuda:0", "cpu"],
                value="mps",
                label="Device",
                info="Device for ColPali model",
            )
            top_k_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=3,
                step=1,
                label="Top-K",
                info="Number of results to retrieve",
            )
        
        # Chat interface
        gr.ChatInterface(
            fn=chat_function,
            additional_inputs=[
                model_dropdown,
                collection_textbox,
                device_dropdown,
                top_k_slider,
            ],
            examples=[
                ["How can I setup encoders of the Faderfox EC-4 MIDI controller such that encoders 2, 3 & 4 control the Reverb of the Digitone II on MIDI channels 6, 7, & 8 respectively? Then, I also want to use encoder 1 to control encoders 2, 3 & 4 simultaneously on the faderfox", "gpt-4o", "midi_manuals", "mps", 5],
                ["What are all the ways to increase decay times in the FM Drum Machine of Digitone II?", "gpt-4o", "midi_manuals", "mps", 5],
                ["What synthesis methods does the Digitone II use?", "gpt-4o-mini", "midi_manuals", "mps", 3],
            ],
            title=None,  # Already have title above
            description=None,  # Already have description above
            chatbot=gr.Chatbot(height=400),
            textbox=gr.Textbox(placeholder="Ask a question about MIDI synthesizers...", scale=7, submit_btn=True),
        )
        
        gr.Markdown(
            "---\n"
            "**Note:** First query may take longer as models are loaded. "
            "Subsequent queries will be faster."
        )
    
    return demo


def main():
    """Launch the Gradio UI."""
    demo = create_ui()
    demo.launch(share=False)


if __name__ == "__main__":
    main()

