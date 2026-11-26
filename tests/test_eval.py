"""Unit tests for evaluation metrics and question schema validation."""

import json
from pathlib import Path

import pytest

from eval.metrics import (
    citation_correctness,
    false_refusal_rate,
    mrr_at_k,
    recall_at_k,
    refusal_recall,
    summarize_metrics,
)
from eval.run_eval import load_eval_questions


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
