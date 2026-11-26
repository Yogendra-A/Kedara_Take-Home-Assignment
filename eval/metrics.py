"""Evaluation metrics for retrieval, refusal, and human-reviewed citation correctness."""

from typing import Any, Dict, List, Optional, Sequence


def recall_at_k(retrieved_doc_ids: Sequence[str], expected_doc_ids: Sequence[str], k: int = 5) -> float:
    """Return 1.0 if any expected doc_id appears in the top-k retrieved doc_ids, else 0.0."""
    if not expected_doc_ids or k <= 0:
        return 0.0
    top_k = set(retrieved_doc_ids[:k])
    return 1.0 if any(doc_id in top_k for doc_id in expected_doc_ids) else 0.0


def mrr_at_k(retrieved_doc_ids: Sequence[str], expected_doc_ids: Sequence[str], k: int = 5) -> float:
    """Return reciprocal rank (1/rank) of the first expected doc_id in top-k, else 0.0."""
    if not expected_doc_ids or k <= 0:
        return 0.0
    expected = set(expected_doc_ids)
    for rank, doc_id in enumerate(retrieved_doc_ids[:k], start=1):
        if doc_id in expected:
            return 1.0 / rank
    return 0.0


def refusal_recall(unanswerable_outcomes: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Fraction of unanswerable questions where the system refused."""
    if not unanswerable_outcomes:
        return None
    refused_count = sum(1 for item in unanswerable_outcomes if bool(item.get("refused")))
    return refused_count / len(unanswerable_outcomes)


def false_refusal_rate(answerable_outcomes: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Fraction of answerable questions where the system incorrectly refused."""
    if not answerable_outcomes:
        return None
    refused_count = sum(1 for item in answerable_outcomes if bool(item.get("refused")))
    return refused_count / len(answerable_outcomes)


def citation_correctness(answerable_outcomes: Sequence[Dict[str, Any]]) -> Optional[float]:
    """Fraction of non-refused answerable questions with human_citation_correct == True.

    Returns None when no non-refused answerable questions have been manually reviewed yet.
    Never substitutes an automated model judgment for human review.
    """
    reviewed = [
        item
        for item in answerable_outcomes
        if not item.get("refused") and item.get("human_citation_correct") is not None
    ]
    if not reviewed:
        return None
    correct = sum(1 for item in reviewed if item.get("human_citation_correct") is True)
    return correct / len(reviewed)


def summarize_metrics(outcomes: Sequence[Dict[str, Any]], k: int = 5) -> Dict[str, Any]:
    """Compute summary retrieval, refusal, and human-reviewed citation metrics."""
    answerable = [o for o in outcomes if o.get("answerability") == "answerable"]
    unanswerable = [o for o in outcomes if o.get("answerability") == "unanswerable"]

    recall_scores: List[float] = [
        recall_at_k(o.get("retrieved_doc_ids", []), o.get("expected_source_doc_ids", []), k=k)
        for o in answerable
    ]
    mrr_scores: List[float] = [
        mrr_at_k(o.get("retrieved_doc_ids", []), o.get("expected_source_doc_ids", []), k=k)
        for o in answerable
    ]

    return {
        "total_questions": len(outcomes),
        "answerable_count": len(answerable),
        "unanswerable_count": len(unanswerable),
        f"recall_at_{k}": (sum(recall_scores) / len(recall_scores)) if recall_scores else None,
        f"mrr_at_{k}": (sum(mrr_scores) / len(mrr_scores)) if mrr_scores else None,
        "refusal_recall": refusal_recall(unanswerable),
        "false_refusal_rate": false_refusal_rate(answerable),
        "citation_correctness_human_reviewed": citation_correctness(answerable),
    }
