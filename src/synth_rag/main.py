import os
import json
from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langgraph import StateGraph, tool, ToolNode, ToolMessage
from langchain.document_loaders import HuggingFaceDatasetLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms import ChatOpenAI
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams
from brave_search import BraveSearch
from pathlib import Path

load_dotenv()
qdrant_key = os.getenv("QDRANT_KEY")
qdrant_url = os.getenv("QDRANT_URL")
brave_key = os.getenv("BRAVE_API_KEY")
