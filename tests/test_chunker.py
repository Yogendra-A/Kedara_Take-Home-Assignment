"""Tests for deterministic token-based chunking and metadata."""

import pytest

from src.chunker import chunk_sections
from src.ingest import DocumentSection


def _make_long_section(doc_id: str, word_count: int) -> DocumentSection:
    text = " ".join(f"word{i}" for i in range(word_count))
    return DocumentSection(
        doc_id=doc_id,
        title=f"Title {doc_id}",
        source_url=f"https://example.gov.in/{doc_id}",
        section="Eligibility",
        text=text,
    )


def test_chunk_metadata_and_unique_ids():
    sections = [
        _make_long_section("doc_a", 300),
        _make_long_section("doc_b", 300),
    ]
    chunks = chunk_sections(sections, chunk_size_tokens=256, chunk_overlap_tokens=32)

    assert len(chunks) >= 4
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids)), "Chunk IDs must be unique"

    for c in chunks:
        assert c.doc_id in {"doc_a", "doc_b"}
        assert c.title == f"Title {c.doc_id}"
        assert c.source_url == f"https://example.gov.in/{c.doc_id}"
        assert c.section == "Eligibility"
        assert 1 <= c.token_count <= 256
        assert c.text


def test_chunk_size_256_vs_512_is_deterministic():
    sec = [_make_long_section("nsap", 600)]
    chunks_256_a = chunk_sections(sec, chunk_size_tokens=256, chunk_overlap_tokens=32)
    chunks_256_b = chunk_sections(sec, chunk_size_tokens=256, chunk_overlap_tokens=32)
    chunks_512 = chunk_sections(sec, chunk_size_tokens=512, chunk_overlap_tokens=32)

    assert chunks_256_a == chunks_256_b
    assert len(chunks_256_a) > len(chunks_512)
    assert all(c.token_count <= 256 for c in chunks_256_a)
    assert all(c.token_count <= 512 for c in chunks_512)


def test_invalid_chunk_params_raise():
    sec = [_make_long_section("nsap", 50)]
    with pytest.raises(ValueError):
        chunk_sections(sec, chunk_size_tokens=0, chunk_overlap_tokens=0)
    with pytest.raises(ValueError):
        chunk_sections(sec, chunk_size_tokens=256, chunk_overlap_tokens=256)
