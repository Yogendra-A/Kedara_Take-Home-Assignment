"""Integration tests for dual FAISS index building (256 and 512 tokens)."""

from pathlib import Path
from typing import List

import numpy as np

from src.index_builder import build_dual_indexes
from src.ingest import DocumentSection
from src.retriever import VectorStore


class DeterministicFakeEmbedder:
    """4-D L2-normalized keyword embedder for fast offline integration tests."""

    KEYWORDS = ("pension", "helpline", "hospital", "tax")

    def encode(self, texts: List[str]) -> np.ndarray:
        vecs = np.zeros((len(texts), len(self.KEYWORDS)), dtype=np.float32)
        for i, text in enumerate(texts):
            lower = text.lower()
            for j, kw in enumerate(self.KEYWORDS):
                if kw in lower:
                    vecs[i, j] += 1.0
            norm = np.linalg.norm(vecs[i])
            if norm > 0:
                vecs[i] /= norm
            else:
                vecs[i, 0] = 1.0
        return vecs


def test_build_dual_indexes_256_and_512_alignment_and_reload(tmp_path: Path):
    sections = [
        DocumentSection(
            doc_id="doc_pension",
            title="IGNOAPS",
            source_url="https://pib.gov.in/pension",
            section="Eligibility",
            text=" ".join(["Senior citizen old age pension scheme."] * 90),
        ),
        DocumentSection(
            doc_id="doc_helpline",
            title="Elderline 14567",
            source_url="https://dosje.gov.in/elderline",
            section="Support",
            text=" ".join(["Toll free elder helpline 14567 for senior citizens."] * 90),
        ),
    ]
    embedder = DeterministicFakeEmbedder()
    stores = build_dual_indexes(
        sections=sections,
        index_base_dir=tmp_path / "index",
        chunk_sizes=(256, 512),
        overlap_tokens=32,
        embedder=embedder,
    )

    assert set(stores.keys()) == {256, 512}
    store_256 = stores[256]
    store_512 = stores[512]

    # Same documents produce more chunks at 256 than at 512
    assert len(store_256.chunks) > len(store_512.chunks)
    assert store_256.index.ntotal == len(store_256.chunks)
    assert store_512.index.ntotal == len(store_512.chunks)

    # Deterministic chunk IDs and multi-document presence
    assert len({c.doc_id for c in store_256.chunks}) == 2
    assert len({c.chunk_id for c in store_256.chunks}) == len(store_256.chunks)

    # Persistence and reload check
    reloaded_256 = VectorStore.load(tmp_path / "index" / "chunks_256", embedder=embedder)
    reloaded_512 = VectorStore.load(tmp_path / "index" / "chunks_512", embedder=embedder)

    res_256 = reloaded_256.search("What is the elder helpline?", top_k=1)
    res_512 = reloaded_512.search("What is the elder helpline?", top_k=1)
    assert res_256[0].chunk.doc_id == "doc_helpline"
    assert res_512[0].chunk.doc_id == "doc_helpline"
    assert 0.99 <= res_256[0].score <= 1.0
