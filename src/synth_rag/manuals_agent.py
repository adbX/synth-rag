#!/usr/bin/env python3
"""
Agentic RAG for PDF manuals using LangGraph, Qdrant, and Brave Search.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Annotated, TypedDict

import torch
from colpali_engine.models import ColPali, ColPaliProcessor
from fastembed import TextEmbedding, SparseTextEmbedding
from langchain_core.messages import ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from qdrant_client import models

from synth_rag.settings import get_qdrant_client, get_api_settings


# System prompt for the agent
SYSTEM_PROMPT = """You are a helpful assistant that answers questions about MIDI synthesizers.

CRITICAL TOOL USAGE RULES:
1. You MUST ALWAYS call manuals_retriever_tool FIRST for every question - no exceptions.
2. After getting results from manuals_retriever_tool, use that information as your PRIMARY source.
3. Only call web_search_tool AFTER you have already called manuals_retriever_tool.
4. If manuals_retriever_tool returns "No relevant information found", then you may use web_search_tool.

RESPONSE FORMAT:
1. Start with "## Information from Manuals" section containing answers based on manual content.
2. ALWAYS cite manual sources in this exact format: (Manual Name, Page X)
3. If you also used web search, add a separate "## Additional Web Search Results" section at the end.

Example citation format:
"The Digitone II has 8 tracks (Digitone-2-User-Manual, Page 12). Each track can be configured independently (Digitone-2-User-Manual, Page 15)."

IMPORTANT: You MUST call manuals_retriever_tool before doing anything else. Do not skip this step.
"""


# State definition for LangGraph
class State(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agentic RAG for PDF manuals")
    parser.add_argument(
        "--question",
        type=str,
        required=True,
        help="Question to ask the agent",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="midi_manuals",
        help="Qdrant collection name",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device for ColPali model (mps, cuda:0, cpu)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of results to retrieve",
    )
    return parser.parse_args()


class ManualsRetriever:
    """Retriever for PDF manuals using hybrid search + ColPali reranking."""
    
    def __init__(self, collection_name: str, device: str, top_k: int = 3):
        self.collection_name = collection_name
        self.device = device
        self.top_k = top_k
        self.client = get_qdrant_client()
        
        # Load models lazily
        self._colpali_model = None
        self._colpali_processor = None
        self._dense_model = None
        self._sparse_model = None
    
    def _load_models(self):
        """Lazy load models on first use."""
        if self._colpali_model is None:
            print("Loading ColPali model...")
            self._colpali_model = ColPali.from_pretrained(
                "vidore/colpali-v1.3",
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            ).eval()
            self._colpali_processor = ColPaliProcessor.from_pretrained("vidore/colpali-v1.3")
            
            print("Loading FastEmbed models...")
            self._dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
            self._sparse_model = SparseTextEmbedding("Qdrant/bm25")
    
    def retrieve(self, query: str) -> list[dict]:
        """Retrieve relevant manual pages for a query."""
        self._load_models()
        
        # Generate embeddings
        dense_embedding = list(self._dense_model.embed([query]))[0]
        sparse_embedding = list(self._sparse_model.embed([query]))[0]
        
        processed_query = self._colpali_processor.process_queries([query]).to(self.device)
        with torch.no_grad():
            query_embedding = self._colpali_model(**processed_query)[0]
        query_embedding_list = query_embedding.cpu().float().numpy().tolist()
        
        # Hybrid search with reranking
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding_list,
            prefetch=[
                models.Prefetch(
                    query=dense_embedding.tolist(),
                    limit=50,
                    using="dense",
                ),
                models.Prefetch(
                    query=sparse_embedding.as_object(),
                    limit=50,
                    using="sparse",
                ),
                models.Prefetch(
                    query=query_embedding_list,
                    limit=50,
                    using="colpali_rows",
                ),
                models.Prefetch(
                    query=query_embedding_list,
                    limit=50,
                    using="colpali_cols",
                ),
            ],
            limit=self.top_k,
            using="colpali_original",
            with_payload=True,
        )
        
        # Format results
        results = []
        for point in response.points:
            results.append({
                "manual_name": point.payload.get("manual_name"),
                "page_num": point.payload.get("page_num"),
                "text": point.payload.get("full_text", point.payload.get("text", "")),
                "score": point.score,
            })
        
        return results


def create_manuals_retriever_tool(retriever: ManualsRetriever):
    """Create a LangChain tool for the manuals retriever."""
    
    @tool("manuals_retriever_tool")
    def retrieve_manuals(query: str) -> str:
        """
        PRIMARY TOOL - MUST be called FIRST for every question.
        Search PDF manuals for MIDI synthesizer information including features, settings, operations, and technical details.
        This tool searches indexed PDF manuals and returns relevant pages with page numbers for citation.
        """
        results = retriever.retrieve(query)
        
        if not results:
            return "No relevant information found in the manuals."
        
        # Format results as text
        output = []
        for i, result in enumerate(results, 1):
            output.append(
                f"[{i}] {result['manual_name']} (Page {result['page_num']}, Score: {result['score']:.3f})\n"
                f"{result['text'][:800]}\n"
            )
        
        return "\n---\n".join(output)
    
    return retrieve_manuals


def create_brave_search_tool():
    """Create a Brave Search tool."""
    from langchain_community.tools import BraveSearch
    
    api_settings = get_api_settings()
    
    @tool("web_search_tool")
    def search_web(query: str) -> str:
        """
        SECONDARY TOOL - Only use AFTER manuals_retriever_tool has been called.
        Search the web for supplementary information about MIDI synthesizers or music production.
        Use only when manuals don't have sufficient information or for general context.
        """
        search = BraveSearch.from_api_key(
            api_key=api_settings.brave_key,
            search_kwargs={"count": 3}
        )
        return search.run(query)
    
    return search_web


def create_agent_graph(
    collection_name: str,
    device: str,
    top_k: int,
    model: str,
):
    """Create the LangGraph agent."""
    
    # Create tools
    retriever = ManualsRetriever(collection_name, device, top_k)
    manuals_tool = create_manuals_retriever_tool(retriever)
    brave_tool = create_brave_search_tool()
    
    tools = [manuals_tool, brave_tool]
    tool_node = ToolNode(tools=tools)
    
    # Create LLM with tools
    api_settings = get_api_settings()
    os.environ["OPENAI_API_KEY"] = api_settings.openai_key
    
    llm = ChatOpenAI(model=model, temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    # Define agent node
    def agent(state: State):
        messages = state["messages"]
        
        # Prepend system message if not already present
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + messages
        
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # Define routing function
    def route(state: State):
        messages = state["messages"]
        last_message = messages[-1]
        
        if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
            return "tools"
        
        return END
    
    # Build graph
    graph_builder = StateGraph(State)
    
    graph_builder.add_node("agent", agent)
    graph_builder.add_node("tools", tool_node)
    
    graph_builder.add_conditional_edges(
        "agent",
        route,
        {"tools": "tools", END: END},
    )
    
    graph_builder.add_edge("tools", "agent")
    graph_builder.add_edge(START, "agent")
    
    return graph_builder.compile()


def run_agent(question: str, collection_name: str, model: str, device: str, top_k: int):
    """Run the agent with a question."""
    print(f"\n{'='*80}")
    print(f"Question: {question}")
    print(f"Collection: {collection_name}")
    print(f"Model: {model}")
    print(f"{'='*80}\n")
    
    # Create and run agent
    graph = create_agent_graph(collection_name, device, top_k, model)
    
    print("Agent is thinking...\n")
    
    for event in graph.stream({"messages": [("user", question)]}):
        for value in event.values():
            last_message = value["messages"][-1]
            
            # Print tool calls
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    print(f"🔧 Using tool: {tool_call['name']}")
                    print(f"   Args: {json.dumps(tool_call['args'], indent=2)}\n")
            
            # Print tool responses
            elif isinstance(last_message, ToolMessage):
                print(f"📄 Tool response from {last_message.name}:")
                content = last_message.content
                if len(content) > 500:
                    print(f"   {content[:500]}...\n")
                else:
                    print(f"   {content}\n")
            
            # Print final answer
            elif hasattr(last_message, "content") and last_message.content:
                if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
                    print(f"\n{'='*80}")
                    print("🤖 Final Answer:")
                    print(f"{'='*80}")
                    print(last_message.content)
                    print(f"{'='*80}\n")


def main():
    args = parse_args()
    
    run_agent(
        question=args.question,
        collection_name=args.collection,
        model=args.model,
        device=args.device,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()

