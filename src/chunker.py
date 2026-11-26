"""Deterministic token-based text chunking with chunk metadata."""

from dataclasses import asdict, dataclass
from typing import List

import tiktoken

from src.ingest import DocumentSection


@dataclass(frozen=True)
class Chunk:
    """A single indexed text chunk with full provenance metadata."""

    chunk_id: str
    doc_id: str
    title: str
    source_url: str
    section: str
    token_count: int
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)


def chunk_sections(
    sections: List[DocumentSection],
    chunk_size_tokens: int = 256,
    chunk_overlap_tokens: int = 32,
    tokenizer_name: str = "cl100k_base",
) -> List[Chunk]:
    """Split DocumentSections into deterministic token-bounded Chunks."""
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive")
    if chunk_overlap_tokens < 0 or chunk_overlap_tokens >= chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be >= 0 and < chunk_size_tokens")

    encoding = tiktoken.get_encoding(tokenizer_name)
    step = chunk_size_tokens - chunk_overlap_tokens
    chunks: List[Chunk] = []
    doc_chunk_counters: dict[str, int] = {}

    for sec in sections:
        token_ids = encoding.encode(sec.text)
        if not token_ids:
            continue

        start = 0
        while start < len(token_ids):
            window = token_ids[start : start + chunk_size_tokens]
            chunk_text = encoding.decode(window).strip()
            if chunk_text:
                idx = doc_chunk_counters.get(sec.doc_id, 0)
                doc_chunk_counters[sec.doc_id] = idx + 1
                chunk_id = f"{sec.doc_id}__c{chunk_size_tokens}__{idx:04d}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=sec.doc_id,
                        title=sec.title,
                        source_url=sec.source_url,
                        section=sec.section,
                        token_count=len(window),
                        text=chunk_text,
                    )
                )
            if start + chunk_size_tokens >= len(token_ids):
                break
            start += step

    return chunks
