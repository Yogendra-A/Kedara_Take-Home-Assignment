"""Embedding, FAISS indexing, persistence, and top-k retrieval."""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List, Protocol

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.chunker import Chunk


class Embedder(Protocol):
    """Protocol for text embedders (allows lightweight fakes in unit tests)."""

    def encode(self, texts: List[str]) -> np.ndarray:
        """Return L2-normalized float32 embeddings of shape (len(texts), dim)."""
        ...


class SentenceTransformerEmbedder:
    """Wrapper around sentence-transformers with L2 normalization."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model: SentenceTransformer | None = None

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vecs = self._get_model().encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved chunk paired with its cosine similarity score."""

    chunk: Chunk
    score: float


def l2_squared_to_similarity(l2_squared: float) -> float:
    """Convert FAISS IndexFlatL2 squared distance of unit vectors to cosine similarity [0, 1]."""
    cos_sim = 1.0 - (l2_squared / 2.0)
    return float(np.clip(cos_sim, 0.0, 1.0))


class VectorStore:
    """FAISS IndexFlatL2 store paired with ordered Chunk metadata."""

    def __init__(
        self,
        index: faiss.IndexFlatL2 | None = None,
        chunks: List[Chunk] | None = None,
        embedder: Embedder | None = None,
    ) -> None:
        self.index = index
        self.chunks: List[Chunk] = chunks or []
        self.embedder: Embedder = embedder or SentenceTransformerEmbedder()

    @property
    def is_empty(self) -> bool:
        return self.index is None or self.index.ntotal == 0 or len(self.chunks) == 0

    def build(self, chunks: List[Chunk]) -> None:
        """Embed chunks and build a fresh FAISS IndexFlatL2."""
        if not chunks:
            self.index = None
            self.chunks = []
            return

        texts = [c.text for c in chunks]
        embeddings = np.ascontiguousarray(self.embedder.encode(texts), dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Embedder returned shape {embeddings.shape}, expected ({len(chunks)}, dim)."
            )
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        self.index = index
        self.chunks = list(chunks)

    def search(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        """Return up to top_k most similar chunks. Returns [] if the corpus is empty."""
        if self.is_empty or not query.strip() or top_k <= 0:
            return []

        q_vec = self.embedder.encode([query]).astype(np.float32)
        k = min(top_k, self.index.ntotal)
        distances, indices = self.index.search(q_vec, k)

        results: List[RetrievedChunk] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[idx],
                    score=round(l2_squared_to_similarity(float(dist)), 4),
                )
            )
        return results

    def save(self, index_dir: Path) -> None:
        """Persist FAISS index and chunk metadata JSONL to index_dir."""
        if self.is_empty or self.index is None:
            raise ValueError("Cannot save an empty VectorStore.")

        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "faiss.index"))
        with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for chunk in self.chunks:
                f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load(
        cls, index_dir: Path, embedder: Embedder | None = None
    ) -> "VectorStore":
        """Load a persisted FAISS index and chunk metadata from index_dir."""
        index_path = index_dir / "faiss.index"
        chunks_path = index_dir / "chunks.jsonl"
        if not index_path.exists() and not chunks_path.exists():
            return cls(index=None, chunks=[], embedder=embedder)
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError(
                f"Incomplete VectorStore in {index_dir}: both faiss.index and chunks.jsonl are required."
            )

        index = faiss.read_index(str(index_path))
        chunks: List[Chunk] = []
        with chunks_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(Chunk.from_dict(json.loads(line)))

        if index.ntotal != len(chunks):
            raise ValueError(
                f"VectorStore mismatch in {index_dir}: index has {index.ntotal} vectors "
                f"but chunks.jsonl has {len(chunks)} chunks."
            )

        return cls(index=index, chunks=chunks, embedder=embedder)
