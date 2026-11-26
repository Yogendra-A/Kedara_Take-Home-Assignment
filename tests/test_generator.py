"""Tests for grounded answer generation, citation validation, refusal, and error handling."""

from dataclasses import dataclass
from typing import List

import pytest

from config.settings import LLMConfig, REFUSAL_TEXT
from src.chunker import Chunk
from src.generator import (
    GeneratorError,
    build_llm_client,
    extract_and_validate_citations,
    generate_grounded_answer,
)


@dataclass(frozen=True)
class FakeRetrievedChunk:
    chunk: Chunk
    score: float


class FakeLLMClient:
    def __init__(self, response: str) -> None:
        self.model = "fake-model"
        self._response = response

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        return self._response


def _sample_retrieved() -> List[FakeRetrievedChunk]:
    return [
        FakeRetrievedChunk(
            chunk=Chunk(
                chunk_id="elderline_14567__c256__0000",
                doc_id="elderline_14567",
                title="Elderline 14567",
                source_url="https://dosje.gov.in/elderline",
                section="Overview",
                token_count=15,
                text="Elderline toll-free helpline number is 14567, open 8 AM to 8 PM.",
            ),
            score=0.88,
        )
    ]


def test_grounded_answer_with_valid_citation():
    retrieved = _sample_retrieved()
    client = FakeLLMClient("The toll-free Elderline number is 14567 [elderline_14567__c256__0000].")
    res = generate_grounded_answer("What is the Elderline number?", retrieved, client=client)

    assert not res.refused
    assert res.citation_valid
    assert res.citations == ["elderline_14567__c256__0000"]
    assert "14567" in res.answer


def test_rejects_fabricated_or_missing_citations():
    retrieved = _sample_retrieved()

    # Fabricated citation ID not in retrieved_chunks
    bad_client = FakeLLMClient("The number is 14567 [fake_doc__c256__0099].")
    res_bad = generate_grounded_answer("What is the number?", retrieved, client=bad_client)
    assert res_bad.refused
    assert not res_bad.citation_valid
    assert res_bad.answer == REFUSAL_TEXT

    # Missing citation altogether
    no_cite_client = FakeLLMClient("The number is 14567.")
    res_none = generate_grounded_answer("What is the number?", retrieved, client=no_cite_client)
    assert res_none.refused
    assert not res_none.citation_valid
    assert res_none.answer == REFUSAL_TEXT


def test_refusal_on_empty_retrieval_or_low_score():
    client = FakeLLMClient("Should not be called")
    res_empty = generate_grounded_answer("Unknown topic?", [], client=client)
    assert res_empty.refused
    assert res_empty.answer == REFUSAL_TEXT

    res_low = generate_grounded_answer(
        "Unknown topic?", _sample_retrieved(), client=client, min_top_score=0.95
    )
    assert res_low.refused
    assert res_low.answer == REFUSAL_TEXT


def test_missing_api_key_and_unsupported_provider_raise():
    with pytest.raises(GeneratorError, match="Missing GROQ_API_KEY"):
        build_llm_client(LLMConfig(provider="groq", groq_api_key=""))

    with pytest.raises(GeneratorError, match="Missing GEMINI_API_KEY"):
        build_llm_client(LLMConfig(provider="gemini", gemini_api_key="your_gemini_api_key_here"))

    with pytest.raises(GeneratorError, match="Unsupported LLM_PROVIDER"):
        build_llm_client(LLMConfig(provider="unknown"))


def test_extract_and_validate_citations_helper():
    retrieved = _sample_retrieved()
    cites, valid = extract_and_validate_citations(
        "Fact [elderline_14567__c256__0000].", retrieved
    )
    assert cites == ["elderline_14567__c256__0000"]
    assert valid is True
