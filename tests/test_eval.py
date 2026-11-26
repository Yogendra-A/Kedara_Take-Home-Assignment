"""Unit tests for evaluation metrics and question schema validation."""

import json
from pathlib import Path

import pytest

from config.settings import REFUSAL_TEXT
from eval.compare import format_comparison_markdown, select_worst_examples
from eval.metrics import (
    citation_correctness,
    false_refusal_rate,
    mrr_at_k,
    recall_at_k,
    refusal_recall,
    summarize_metrics,
)
from eval.run_eval import evaluate_index, load_eval_questions
from src.chunker import Chunk
from src.generator import GeneratorError
from src.ingest import DocumentSection
from src.index_builder import build_dual_indexes
from tests.test_index_builder import DeterministicFakeEmbedder


def test_recall_and_mrr_at_k():
    retrieved = ["doc_a", "doc_b", "doc_c", "doc_d", "doc_e"]
    assert recall_at_k(retrieved, ["doc_c"], k=5) == 1.0
    assert recall_at_k(retrieved, ["doc_z"], k=5) == 0.0
    assert mrr_at_k(retrieved, ["doc_a"], k=5) == 1.0
    assert mrr_at_k(retrieved, ["doc_c"], k=5) == pytest.approx(1.0 / 3.0)
    assert mrr_at_k(retrieved, ["doc_z"], k=5) == 0.0


def test_refusal_and_human_citation_metrics():
    outcomes = [
        {
            "answerability": "answerable",
            "retrieved_doc_ids": ["doc_a", "doc_b"],
            "expected_source_doc_ids": ["doc_b"],
            "refused": False,
            "human_citation_correct": True,
        },
        {
            "answerability": "answerable",
            "retrieved_doc_ids": ["doc_x"],
            "expected_source_doc_ids": ["doc_y"],
            "refused": True,
            "human_citation_correct": None,
        },
        {
            "answerability": "unanswerable",
            "retrieved_doc_ids": ["doc_x"],
            "expected_source_doc_ids": [],
            "refused": True,
            "human_citation_correct": None,
        },
        {
            "answerability": "unanswerable",
            "retrieved_doc_ids": ["doc_x"],
            "expected_source_doc_ids": [],
            "refused": False,
            "human_citation_correct": None,
        },
    ]
    summary = summarize_metrics(outcomes, k=5)
    assert summary["recall_at_5"] == pytest.approx(0.5)
    assert summary["mrr_at_5"] == pytest.approx(0.25)
    assert summary["refusal_recall"] == pytest.approx(0.5)
    assert summary["false_refusal_rate"] == pytest.approx(0.5)
    assert summary["citation_correctness_human_reviewed"] == pytest.approx(1.0)


def test_citation_correctness_returns_none_when_unreviewed():
    unreviewed = [
        {"answerability": "answerable", "refused": False, "human_citation_correct": None}
    ]
    assert citation_correctness(unreviewed) is None
    assert refusal_recall([]) is None
    assert false_refusal_rate([]) is None


def test_load_eval_questions_validates_schema(tmp_path: Path):
    q_path = tmp_path / "questions.json"
    q_path.write_text(
        json.dumps([
            {
                "question_id": "Q1",
                "question_text": "What is Elderline?",
                "answerability": "answerable",
                "question_category": "helpline",
                "expected_source_doc_ids": ["elderline_14567"],
                "expected_answer_facts": ["14567"],
            }
        ]),
        encoding="utf-8",
    )
    loaded = load_eval_questions(q_path)
    assert len(loaded) == 1

    bad_path = tmp_path / "bad.json"
    bad_path.write_text(json.dumps([{"question_id": "Q1"}]), encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        load_eval_questions(bad_path)

    empty_docs_path = tmp_path / "empty_docs.json"
    empty_docs_path.write_text(
        json.dumps([
            {
                "question_id": "Q2",
                "question_text": "What is Elderline?",
                "answerability": "answerable",
                "question_category": "helpline",
                "expected_source_doc_ids": [],
                "expected_answer_facts": ["14567"],
            }
        ]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="at least one expected_source_doc_id"):
        load_eval_questions(empty_docs_path)

    unans_with_docs = tmp_path / "unans_docs.json"
    unans_with_docs.write_text(
        json.dumps([
            {
                "question_id": "Q3",
                "question_text": "What is US Medicare?",
                "answerability": "unanswerable",
                "question_category": "out_of_corpus",
                "expected_source_doc_ids": ["elderline_14567"],
                "expected_answer_facts": [],
            }
        ]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="empty expected_source_doc_ids"):
        load_eval_questions(unans_with_docs)

    with pytest.raises(ValueError, match="unknown doc_id"):
        load_eval_questions(q_path, valid_doc_ids={"other_doc"})


class _SequenceLLM:
    def __init__(self, responses):
        self.model = "fake-eval-llm"
        self._responses = list(responses)
        self._idx = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        val = self._responses[min(self._idx, len(self._responses) - 1)]
        self._idx += 1
        if isinstance(val, Exception):
            raise val
        return val


def test_evaluate_index_separates_failures_and_preserves_none_human_labels(tmp_path: Path):
    sections = [
        DocumentSection(
            doc_id="doc_helpline",
            title="Elderline 14567",
            source_url="https://dosje.gov.in/elderline",
            section="Support",
            text=" ".join(["Toll free elder helpline 14567 for senior citizens."] * 40),
        ),
        DocumentSection(
            doc_id="doc_pension",
            title="IGNOAPS",
            source_url="https://pib.gov.in/pension",
            section="Eligibility",
            text=" ".join(["Senior citizen old age pension scheme IGNOAPS."] * 40),
        ),
    ]
    embedder = DeterministicFakeEmbedder()
    build_dual_indexes(
        sections=sections,
        index_base_dir=tmp_path / "index",
        chunk_sizes=(256,),
        overlap_tokens=32,
        embedder=embedder,
    )

    questions = [
        {
            "question_id": "Q1",
            "question_text": "What is the elder helpline?",
            "answerability": "answerable",
            "question_category": "helpline",
            "expected_source_doc_ids": ["doc_helpline"],
            "expected_answer_facts": ["14567"],
        },
        {
            "question_id": "Q2",
            "question_text": "What is the pension scheme?",
            "answerability": "answerable",
            "question_category": "pension",
            "expected_source_doc_ids": ["doc_pension"],
            "expected_answer_facts": ["IGNOAPS"],
        },
        {
            "question_id": "Q3",
            "question_text": "What is the hospital benefit?",
            "answerability": "answerable",
            "question_category": "healthcare",
            "expected_source_doc_ids": ["doc_pension"],
            "expected_answer_facts": ["pension"],
        },
        {
            "question_id": "Q4",
            "question_text": "What is the tax rate in Japan?",
            "answerability": "unanswerable",
            "question_category": "out_of_corpus",
            "expected_source_doc_ids": [],
            "expected_answer_facts": [],
        },
    ]

    # Q1: missing citations twice -> missing_citations
    # Q2: malformed handle [S9] twice -> malformed_citations
    # Q3: provider error -> provider_error (not counted as refusal)
    # Q4: explicit refusal -> refused=True
    llm = _SequenceLLM([
        "The helpline is 14567.",
        "Still no citation.",
        "Pension is IGNOAPS [S9].",
        "Still bad [S99].",
        GeneratorError("Rate limit 429"),
        REFUSAL_TEXT,
    ])

    report = evaluate_index(
        questions=questions,
        chunk_size=256,
        top_k=5,
        min_score=0.0,
        index_dir=tmp_path / "index" / "chunks_256",
        embedder=embedder,
        llm_client=llm,
    )
    summary = report["summary"]
    assert summary["answerable_count"] == 3
    assert summary["unanswerable_count"] == 1
    assert summary["missing_citations_count"] == 1
    assert summary["malformed_citations_count"] == 1
    assert summary["provider_error_count"] == 1
    assert summary["refusal_recall"] == pytest.approx(1.0)
    assert summary["false_refusal_rate"] == pytest.approx(0.0)
    assert summary["citation_correctness_human_reviewed"] is None
    assert all(o["human_citation_correct"] is None for o in report["outcomes"])

    worst = select_worst_examples(report["outcomes"], limit=2)
    assert len(worst) == 2
    assert worst[0]["error_type"] == "provider_error"

    md = format_comparison_markdown(
        {
            "settings": {"embedding_model": "all-MiniLM-L6-v2", "top_k": 5, "min_score": 0.2},
            "summary_256": summary,
            "summary_512": summary,
            "results_256": report,
        }
    )
    assert "256 vs 512 Comparison Table" in md
    assert "Five Worst-Answer Examples" in md
