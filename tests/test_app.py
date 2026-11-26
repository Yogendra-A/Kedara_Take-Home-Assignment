"""Tests for Streamlit UI data transformations, latency formatting, and error handling."""

from pathlib import Path
from typing import List

import pytest

from config.settings import LLMConfig, REFUSAL_TEXT
from src.chunker import Chunk
from src.generator import GeneratorError
from src.index_builder import build_dual_indexes
from src.ingest import DocumentSection
from src.retriever import VectorStore
from tests.test_index_builder import DeterministicFakeEmbedder

from app import (
    build_citation_rows,
    build_evidence_rows,
    execute_rag_query,
    format_latency,
    get_corpus_doc_count,
    sanitize_error_message,
)


class _StubLLM:
    def __init__(self, response: str | Exception) -> None:
        self.model = "stub-model"
        self._response = response

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def _build_test_store(tmp_path: Path) -> VectorStore:
    sections = [
        DocumentSection(
            doc_id="elderline_14567",
            title="Elderline 14567",
            source_url="https://www.dosje.gov.in/elderline",
            section="Overview",
            text=" ".join(["Elderline toll free helpline 14567 for senior citizens."] * 30),
        ),
        DocumentSection(
            doc_id="doc_pension",
            title="IGNOAPS Pension",
            source_url="https://pib.gov.in/pension",
            section="Eligibility",
            text=" ".join(["Senior citizen old age pension scheme IGNOAPS."] * 30),
        ),
    ]
    embedder = DeterministicFakeEmbedder()
    stores = build_dual_indexes(
        sections=sections,
        index_base_dir=tmp_path / "index",
        chunk_sizes=(256,),
        overlap_tokens=32,
        embedder=embedder,
    )
    return stores[256]


def test_format_latency_and_sanitize_error():
    assert format_latency(42.34) == "42.3 ms"
    assert format_latency(1250.0) == "1.25 s (1250 ms)"
    assert format_latency(-5.0) == "0.0 ms"

    leaked = "Auth failed with key gsk_1234567890abcdef and Bearer secret_token_xyz"
    sanitized = sanitize_error_message(leaked)
    assert "gsk_1234567890abcdef" not in sanitized
    assert "secret_token_xyz" not in sanitized
    assert "[REDACTED]" in sanitized


def test_execute_rag_query_answerable_and_citations(tmp_path: Path):
    store = _build_test_store(tmp_path)
    assert get_corpus_doc_count(store) == 2

    outcome = execute_rag_query(
        question="What is the helpline number?",
        store=store,
        top_k=2,
        min_score=0.0,
        llm_client=_StubLLM("The toll-free Elderline helpline is 14567 [S1]."),
        llm_config=LLMConfig(provider="groq"),
    )
    assert outcome.error_message is None
    assert outcome.grounded_answer is not None
    assert not outcome.grounded_answer.refused
    assert outcome.grounded_answer.citation_valid
    assert outcome.retrieval_ms >= 0.0
    assert outcome.generation_ms >= 0.0
    assert outcome.total_ms == pytest.approx(outcome.retrieval_ms + outcome.generation_ms)

    citations = build_citation_rows(outcome.grounded_answer)
    assert len(citations) == 1
    assert citations[0]["handle"] == "S1"
    assert citations[0]["doc_id"] == "elderline_14567"
    assert citations[0]["source_url"] == "https://www.dosje.gov.in/elderline"

    evidence = build_evidence_rows(outcome.retrieved_chunks)
    assert len(evidence) == 2
    assert evidence[0]["rank"] == 1


def test_execute_rag_query_refusal_and_error_handling(tmp_path: Path):
    store = _build_test_store(tmp_path)

    # Genuine refusal
    refusal_outcome = execute_rag_query(
        question="Does Ayushman cover dental implants in the US?",
        store=store,
        top_k=2,
        min_score=0.0,
        llm_client=_StubLLM(REFUSAL_TEXT),
    )
    assert refusal_outcome.error_message is None
    assert refusal_outcome.grounded_answer is not None
    assert refusal_outcome.grounded_answer.refused
    assert refusal_outcome.grounded_answer.answer == REFUSAL_TEXT

    # Provider error handled without crashing and without exposing keys
    err_outcome = execute_rag_query(
        question="What is the helpline?",
        store=store,
        top_k=2,
        min_score=0.0,
        llm_client=_StubLLM(GeneratorError("Groq API error with gsk_secret123456789")),
    )
    assert err_outcome.grounded_answer is None
    assert err_outcome.error_message is not None
    assert "gsk_secret123456789" not in err_outcome.error_message
    assert "[REDACTED]" in err_outcome.error_message

    # Empty store handled gracefully
    empty_store = VectorStore(index=None, chunks=[])
    assert get_corpus_doc_count(empty_store) == 0
    empty_outcome = execute_rag_query("Any question?", store=empty_store)
    assert empty_outcome.error_message is not None
    assert "index_builder" in empty_outcome.error_message
