"""Central configuration for Ask-a-Care-Resource RAG pipeline."""

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
MANIFEST_PATH = DATA_DIR / "manifest.json"

REFUSAL_TEXT = "I don't know from these sources"


@dataclass(frozen=True)
class RetrievalConfig:
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    chunk_size_tokens: int = 256
    chunk_overlap_tokens: int = 32
    top_k: int = int(os.getenv("TOP_K", "5"))
    tokenizer_name: str = "cl100k_base"


@dataclass(frozen=True)
class LLMConfig:
    provider: str = os.getenv("LLM_PROVIDER", "groq")
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    temperature: float = 0.0
    max_tokens: int = 512


def get_index_dir(chunk_size_tokens: int) -> Path:
    """Return the index directory for a specific token chunk size."""
    return INDEX_DIR / f"chunks_{chunk_size_tokens}"
