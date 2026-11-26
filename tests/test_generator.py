"""Tests for grounded answer generation, S1..Sn citation resolution, failure categories, and error handling."""

from dataclasses import dataclass
from typing import List

import pytest

from config.settings import LLMConfig, REFUSAL_TEXT
from src.chunker import Chunk
from src.generator import (
    GeneratorError,
    build_handle_map,
    build_llm_client,
    generate_grounded_answer,
    resolve_handle_citations,
)


@dataclass(frozen=True)
class FakeRetrievedChunk:
    chunk: Chunk
    score: float


class FakeLLMClient:
    def __init__(self, responses: List[str] | str) -> None:
        self.model = "fake-model"
        self._responses = [responses] if isinstance(responses, str) else list(responses)
        self.calls = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        idx = min(self.calls, len(self._responses) - 1)
        self.calls += 1
        return self._responses[idx]


class FailingLLMClient:
    def __init__(self) -> None:
        self.model = "failing-model"

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise GeneratorError("Groq rate limit exceeded: 429")


def _sample_retrieved() -> List[FakeRetrievedChunk]:
    return [
        FakeRetrievedChunk(
            chunk=Chunk(
                chunk_id="elderline_14567__c256__0000",
                doc_id="elderline_14567",
                title="Elderline 14567",
                source_url="https://www.dosje.gov.in/elderline",
                section="Overview",
                token_count=15,
                text="Elderline toll-free helpline number is 14567, open 8 AM to 8 PM.",
            ),
            score=0.88,
        ),
        FakeRetrievedChunk(
            chunk=Chunk(
                chunk_id="rbi_senior__c256__0001",
                doc_id="rbi_senior",
                title="RBI Senior Facilities",
                source_url="https://www.rbi.org.in/senior",
                section="Doorstep Banking",
                token_count=14,
                text="Banks provide doorstep banking for senior citizens aged over 70.",
            ),
            score=0.72,
        ),
    ]


def test_short_handle_mapping_and_provenance_resolution():
    retrieved = _sample_retrieved()
    handle_map = build_handle_map(retrieved)
    assert list(handle_map.keys()) == ["S1", "S2"]

    client = FakeLLMClient("The toll-free Elderline number is 14567 [S1].")
    res = generate_grounded_answer("What is the Elderline number?", retrieved, client=client)

    assert not res.refused
    assert res.citation_valid
    assert res.error_type == "none"
    assert res.cited_handles == ["S1"]
    assert res.citations == ["elderline_14567__c256__0000"]
    assert len(res.citation_provenance) == 1
    assert res.citation_provenance[0]["source_url"] == "https://www.dosje.gov.in/elderline"
    assert res.citation_provenance[0]["doc_id"] == "elderline_14567"


def test_unknown_and_missing_citations_trigger_retry_and_malformed_category():
    retrieved = _sample_retrieved()

    # Unknown handle [S9] on both initial and retry -> malformed_citations, NOT refused
    bad_client = FakeLLMClient(["The number is 14567 [S9].", "Still wrong [S99]."])
    res_bad = generate_grounded_answer("What is the number?", retrieved, client=bad_client)
    assert not res_bad.refused
    assert not res_bad.citation_valid
    assert res_bad.error_type == "malformed_citations"
    assert res_bad.initial_failure == "malformed_citations"
    assert bad_client.calls == 2

    # Missing citation initially, fixed on constrained retry -> succeeds while preserving initial_failure
    retry_fix_client = FakeLLMClient(["The number is 14567.", "The number is 14567 [S1]."])
    res_fixed = generate_grounded_answer("What is the number?", retrieved, client=retry_fix_client)
    assert not res_fixed.refused
    assert res_fixed.citation_valid
    assert res_fixed.error_type == "none"
    assert res_fixed.initial_failure == "malformed_citations"
    assert res_fixed.citations == ["elderline_14567__c256__0000"]


def test_legitimate_refusal_on_insufficient_evidence():
    retrieved = _sample_retrieved()

    # Empty retrieval
    res_empty = generate_grounded_answer("Unknown?", [], client=FakeLLMClient("ignored"))
    assert res_empty.refused
    assert res_empty.error_type == "evidence_insufficient"
    assert res_empty.answer == REFUSAL_TEXT

    # Low similarity score
    res_low = generate_grounded_answer(
        "Unknown?", retrieved, client=FakeLLMClient("ignored"), min_top_score=0.95
    )
    assert res_low.refused
    assert res_low.error_type == "evidence_insufficient"
    assert res_low.answer == REFUSAL_TEXT

    # Model explicitly refuses
    res_model_refusal = generate_grounded_answer(
        "Unknown?", retrieved, client=FakeLLMClient(REFUSAL_TEXT)
    )
    assert res_model_refusal.refused
    assert res_model_refusal.error_type == "evidence_insufficient"
    assert res_model_refusal.answer == REFUSAL_TEXT


def test_provider_errors_not_counted_as_refusals():
    with pytest.raises(GeneratorError, match="Missing GROQ_API_KEY"):
        build_llm_client(LLMConfig(provider="groq", groq_api_key=""))

    with pytest.raises(GeneratorError, match="Missing GEMINI_API_KEY"):
        build_llm_client(LLMConfig(provider="gemini", gemini_api_key="your_gemini_api_key_here"))

    with pytest.raises(GeneratorError, match="Unsupported LLM_PROVIDER"):
        build_llm_client(LLMConfig(provider="unknown"))

    with pytest.raises(GeneratorError, match="rate limit"):
        generate_grounded_answer("What is Elderline?", _sample_retrieved(), client=FailingLLMClient())


def test_resolve_handle_citations_helper():
    handle_map = build_handle_map(_sample_retrieved())
    handles, chunk_ids, prov, valid = resolve_handle_citations("Fact [S1] and [S2].", handle_map)
    assert handles == ["S1", "S2"]
    assert chunk_ids == ["elderline_14567__c256__0000", "rbi_senior__c256__0001"]
    assert len(prov) == 2
    assert valid is True
