from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal
import shutil

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()


REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "documents"
MIDI_DIR = DOCS_DIR / "midi_synthesizers"
INPUT_DIR = MIDI_DIR / "input"
TMP_DIR = MIDI_DIR / "tmp"
TMP_PAGES_DIR = TMP_DIR / "pages"
TMP_TEXT_DIR = TMP_DIR / "text"
LOGS_DIR = REPO_ROOT / "logs"
MANUAL_QUERY_LOGS_DIR = LOGS_DIR / "manuals_queries"
BENCHMARK_LOGS_DIR = LOGS_DIR / "benchmark_ragbench"


@dataclass(frozen=True)
class APISettings:
    qdrant_url: str
    qdrant_key: str
    openai_key: str
    brave_key: str


def _get_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


@lru_cache(maxsize=1)
def get_api_settings() -> APISettings:
    return APISettings(
        qdrant_url=_get_env_var("QDRANT_URL"),
        qdrant_key=_get_env_var("QDRANT_KEY"),
        openai_key=_get_env_var("OPENAI_API_KEY"),
        brave_key=_get_env_var("BRAVE_API_KEY"),
    )


@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    api = get_api_settings()
    return QdrantClient(url=api.qdrant_url, api_key=api.qdrant_key, timeout=60)


def get_manual_input_dir(subset: Literal["test", "full"]) -> Path:
    target = INPUT_DIR / subset
    if not target.exists():
        raise FileNotFoundError(f"Manual subset directory not found: {target}")
    return target


def ensure_tmp_dirs(clear: bool = False) -> tuple[Path, Path]:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    TMP_PAGES_DIR.mkdir(parents=True, exist_ok=True)
    TMP_TEXT_DIR.mkdir(parents=True, exist_ok=True)
    if clear:
        for path in (TMP_PAGES_DIR, TMP_TEXT_DIR):
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
    return TMP_PAGES_DIR, TMP_TEXT_DIR


def ensure_logs_dir() -> Path:
    MANUAL_QUERY_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    return MANUAL_QUERY_LOGS_DIR


def ensure_benchmark_logs_dir(dataset_name: str) -> Path:
    """Ensure benchmark logs directory exists for a specific dataset."""
    dataset_logs_dir = BENCHMARK_LOGS_DIR / dataset_name
    dataset_logs_dir.mkdir(parents=True, exist_ok=True)
    return dataset_logs_dir


# RAGBench dataset names
RAGBENCH_DATASETS = [
    "emanual",
    "covidqa",
    "cuad",
    "delucionqa",
    "expertqa",
    "finqa",
    "hagrid",
    "hotpotqa",
    "msmarco",
    "pubmedqa",
    "tatqa",
    "techqa",
]

