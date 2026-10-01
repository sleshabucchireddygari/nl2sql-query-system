"""Central settings, read from environment variables / a .env file."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_SQLITE_URL = f"sqlite:///{PROJECT_ROOT / 'data' / 'hospital.db'}"
DEFAULT_DESCRIPTIONS = PROJECT_ROOT / "data" / "schema_descriptions.json"


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    database_url: str = field(default_factory=lambda: os.getenv("DATABASE_URL") or DEFAULT_SQLITE_URL)
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini").strip().lower())
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    gemini_embed_model: str = field(default_factory=lambda: os.getenv("GEMINI_EMBED_MODEL", "gemini-embedding-001"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b"))
    ollama_embed_model: str = field(default_factory=lambda: os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "text-embedding-3-small"))
    max_retries: int = field(default_factory=lambda: _int("MAX_RETRIES", 2))
    table_top_k: int = field(default_factory=lambda: _int("TABLE_TOP_K", 4))
    row_limit: int = field(default_factory=lambda: _int("ROW_LIMIT", 500))
    descriptions_path: Path = DEFAULT_DESCRIPTIONS
