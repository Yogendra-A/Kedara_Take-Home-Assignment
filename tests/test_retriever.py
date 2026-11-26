"""Tests for FAISS VectorStore, multi-document indexing, mappings, and empty corpus."""

from pathlib import Path
from typing import List

import numpy as np
import pytest

from src.chunker import Chunk
from src.retriever import VectorStore


class KeywordFakeEmbedder:
    """Deterministic 4-D L2-normalized embedder for fast offline retrieval tests."""

    KEYWORDS = ("pension", "helpline", "hospital", "maintenance")

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


def _sample_chunks() -> List[Chunk]:
    return [
        Chunk(
            chunk_id="doc1__c256__0000",
            doc_id="doc1",
            title="Old Age Pension",
            source_url="https://nsap.nic.in/pension",
            section="Amount",
            token_count=12,
            text="Monthly pension for senior citizens under IGNOAPS.",
        ),
        Chunk(
            chunk_id="doc2__c256__0000",
            doc_id="doc2",
            title="Elderline 14567",
            source_url="https://elderline.dosje.gov.in/",
            section="Helpline",
            token_count=11,
            text="National helpline 14567 assists elderly parents in distress.",
        ),
        Chunk(
            chunk_id="doc3__c256__0000",
            doc_id="doc3",
            title="NPHCE Healthcare",
            source_url="https://mohfw.gov.in/nphce",
            section="Services",
            token_count=10,
            text="Dedicated geriatric ward in district hospital.",
        ),
    ]


def test_multi_document_indexing_and_retrieval_mapping():
    store = VectorStore(embedder=KeywordFakeEmbedder())
    store.build(_sample_chunks())

    results = store.search("What is the elder helpline number?", top_k=2)
    assert len(results) == 2
    top = results[0]
    assert top.chunk.doc_id == "doc2"
    assert top.chunk.chunk_id == "doc2__c256__0000"
    assert top.chunk.source_url == "https://elderline.dosje.gov.in/"
    assert top.score == pytest.approx(1.0, abs=1e-4)


def test_empty_corpus_handling(tmp_path: Path):
    store = VectorStore(embedder=KeywordFakeEmbedder())
    assert store.is_empty
    assert store.search("pension", top_k=5) == []

    store.build([])
    assert store.is_empty
    assert store.search("pension", top_k=5) == []

    with pytest.raises(ValueError):
        store.save(tmp_path / "empty_idx")

    loaded = VectorStore.load(tmp_path / "does_not_exist", embedder=KeywordFakeEmbedder())
    assert loaded.is_empty
    assert loaded.search("pension", top_k=5) == []


def test_save_and_load_preserves_mappings(tmp_path: Path):
    embedder = KeywordFakeEmbedder()
    store = VectorStore(embedder=embedder)
    store.build(_sample_chunks())

    idx_dir = tmp_path / "chunks_256"
    store.save(idx_dir)

    reloaded = VectorStore.load(idx_dir, embedder=embedder)
    assert not reloaded.is_empty
    results = reloaded.search("district hospital geriatric", top_k=1)
    assert len(results) == 1
    assert results[0].chunk.doc_id == "doc3"
    assert results[0].chunk.chunk_id == "doc3__c256__0000"


def test_load_raises_on_incomplete_or_mismatched_store(tmp_path: Path):
    embedder = KeywordFakeEmbedder()
    store = VectorStore(embedder=embedder)
    store.build(_sample_chunks())

    idx_dir = tmp_path / "corrupt_idx"
    store.save(idx_dir)

    # Truncate chunks.jsonl so metadata count != index.ntotal
    chunks_path = idx_dir / "chunks.jsonl"
    first_line = chunks_path.read_text(encoding="utf-8").splitlines()[0]
    chunks_path.write_text(first_line + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        VectorStore.load(idx_dir, embedder=embedder)

    # Remove chunks.jsonl completely while leaving faiss.index
    chunks_path.unlink()
    with pytest.raises(FileNotFoundError, match="Incomplete VectorStore"):
        VectorStore.load(idx_dir, embedder=embedder)


def test_build_raises_on_embedder_shape_mismatch():
    class BadEmbedder:
        def encode(self, texts: List[str]) -> np.ndarray:
            return np.zeros((1, 4), dtype=np.float32)

    store = VectorStore(embedder=BadEmbedder())
    with pytest.raises(ValueError, match="Embedder returned shape"):
        store.build(_sample_chunks())
