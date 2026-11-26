"""Evaluation runner for a single FAISS index configuration."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

from config.settings import LLMConfig, RetrievalConfig, get_index_dir
from eval.metrics import summarize_metrics
from src.generator import GeneratorError, LLMClient, generate_grounded_answer
from src.retriever import Embedder, SentenceTransformerEmbedder, VectorStore

REQUIRED_QUESTION_FIELDS = {
    "question_id",
    "question_text",
    "answerability",
    "question_category",
    "expected_source_doc_ids",
    "expected_answer_facts",
}


def load_eval_questions(
    questions_path: Path, valid_doc_ids: set[str] | None = None
) -> List[Dict[str, Any]]:
    """Load and validate evaluation questions from JSON."""
    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")

    items = json.loads(questions_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise ValueError("Questions file must contain a JSON list.")

    seen_ids: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            raise ValueError(f"Each question must be a JSON object: {item}")
        missing = REQUIRED_QUESTION_FIELDS - set(item.keys())
        if missing:
            raise ValueError(f"Question {item.get('question_id')} missing fields: {sorted(missing)}")
        qid = str(item["question_id"]).strip()
        if not qid or qid in seen_ids:
            raise ValueError(f"Invalid or duplicate question_id: {qid!r}")
        seen_ids.add(qid)

        qtext = item["question_text"]
        if not isinstance(qtext, str) or not qtext.strip():
            raise ValueError(f"Question {qid} must have a non-empty string question_text.")

        qcat = item["question_category"]
        if not isinstance(qcat, str) or not qcat.strip():
            raise ValueError(f"Question {qid} must have a non-empty string question_category.")

        answerability = item["answerability"]
        if answerability not in {"answerable", "unanswerable"}:
            raise ValueError(
                f"Question {qid} has invalid answerability: {answerability!r}"
            )

        expected_docs = item["expected_source_doc_ids"]
        if not isinstance(expected_docs, list) or not all(
            isinstance(d, str) and d.strip() for d in expected_docs
        ):
            raise ValueError(
                f"Question {qid} expected_source_doc_ids must be a list of non-empty strings."
            )

        expected_facts = item["expected_answer_facts"]
        if not isinstance(expected_facts, list) or not all(
            isinstance(f, str) and f.strip() for f in expected_facts
        ):
            raise ValueError(
                f"Question {qid} expected_answer_facts must be a list of non-empty strings."
            )

        if answerability == "answerable":
            if not expected_docs:
                raise ValueError(
                    f"Answerable question {qid} must specify at least one expected_source_doc_id."
                )
            if not expected_facts:
                raise ValueError(
                    f"Answerable question {qid} must specify at least one expected_answer_fact."
                )
        elif expected_docs:
            raise ValueError(
                f"Unanswerable question {qid} must have empty expected_source_doc_ids."
            )

        if valid_doc_ids is not None:
            unknown = [d for d in expected_docs if d not in valid_doc_ids]
            if unknown:
                raise ValueError(
                    f"Question {qid} references unknown doc_id(s): {unknown}"
                )
    return items


def _classify_failure_category(
    refused: bool,
    citation_valid: bool,
    error_type: str,
    cited_handles: List[str],
) -> str:
    """Separate missing citations, malformed citation handles, and provider errors."""
    if error_type == "provider_error":
        return "provider_error"
    if refused:
        return "none"
    if not citation_valid:
        return "missing_citations" if not cited_handles else "malformed_citations"
    return "none"


def evaluate_index(
    questions: List[Dict[str, Any]],
    chunk_size: int = 256,
    top_k: int = 5,
    min_score: float = 0.2,
    index_dir: Path | None = None,
    embedder: Embedder | None = None,
    llm_client: LLMClient | None = None,
    llm_config: LLMConfig | None = None,
) -> Dict[str, Any]:
    """Run retrieval and grounded generation over questions and compute summary metrics."""
    cfg = RetrievalConfig()
    target_dir = index_dir or get_index_dir(chunk_size)
    active_embedder = embedder or SentenceTransformerEmbedder(model_name=cfg.embedding_model)
    store = VectorStore.load(target_dir, embedder=active_embedder)
    if store.is_empty:
        raise RuntimeError(f"Index at {target_dir} is empty or missing.")

    outcomes: List[Dict[str, Any]] = []
    for q in questions:
        retrieved = store.search(q["question_text"], top_k=top_k)
        try:
            result = generate_grounded_answer(
                question=q["question_text"],
                retrieved_chunks=retrieved,
                client=llm_client,
                config=llm_config,
                min_top_score=min_score,
            )
            answer = result.answer
            citations = result.citations
            cited_handles = result.cited_handles
            citation_provenance = result.citation_provenance
            refused = result.refused
            citation_valid = result.citation_valid
            error_type = result.error_type
            initial_failure = result.initial_failure
        except GeneratorError as exc:
            answer = f"PROVIDER_ERROR: {exc}"
            citations = []
            cited_handles = []
            citation_provenance = []
            refused = False
            citation_valid = False
            error_type = "provider_error"
            initial_failure = "provider_error"

        failure_category = _classify_failure_category(
            refused=refused,
            citation_valid=citation_valid,
            error_type=error_type,
            cited_handles=cited_handles,
        )

        outcomes.append(
            {
                "question_id": q["question_id"],
                "question_text": q["question_text"],
                "answerability": q["answerability"],
                "question_category": q["question_category"],
                "expected_source_doc_ids": q["expected_source_doc_ids"],
                "expected_answer_facts": q["expected_answer_facts"],
                "retrieved_doc_ids": [rc.chunk.doc_id for rc in retrieved],
                "retrieved_chunk_ids": [rc.chunk.chunk_id for rc in retrieved],
                "retrieved_scores": [rc.score for rc in retrieved],
                "retrieved_evidence": [
                    {**asdict(rc.chunk), "score": rc.score} for rc in retrieved
                ],
                "answer": answer,
                "citations": citations,
                "cited_handles": cited_handles,
                "citation_provenance": citation_provenance,
                "refused": refused,
                "citation_valid": citation_valid,
                "error_type": error_type,
                "failure_category": failure_category,
                "initial_failure": initial_failure,
                "human_citation_correct": None,
            }
        )

    summary = summarize_metrics(outcomes, k=top_k)
    return {
        "chunk_size": chunk_size,
        "top_k": top_k,
        "min_score": min_score,
        "summary": summary,
        "outcomes": outcomes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Kedara RAG evaluation on a single index.")
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.json"))
    parser.add_argument("--chunk-size", type=int, default=256, choices=[256, 512])
    parser.add_argument("--top-k", type=int, default=RetrievalConfig().top_k)
    parser.add_argument("--min-score", type=float, default=0.2)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate evaluation JSON schema and doc_ids against the corpus without running the LLM.",
    )
    args = parser.parse_args()

    valid_doc_ids: set[str] | None = None
    sections_path = Path("data/processed/sections.jsonl")
    if sections_path.exists():
        valid_doc_ids = {
            json.loads(line)["doc_id"]
            for line in sections_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    questions = load_eval_questions(args.questions, valid_doc_ids=valid_doc_ids)
    if not questions:
        print(
            f"No questions found in {args.questions}. "
            "Populate it manually using eval/questions.template.json before running evaluation.",
            file=sys.stderr,
        )
        return 1

    if args.validate_only:
        ans = sum(1 for q in questions if q["answerability"] == "answerable")
        unans = sum(1 for q in questions if q["answerability"] == "unanswerable")
        print(
            f"Validated {len(questions)} questions in {args.questions} "
            f"({ans} answerable, {unans} unanswerable)."
        )
        return 0

    report = evaluate_index(
        questions=questions,
        chunk_size=args.chunk_size,
        top_k=args.top_k,
        min_score=args.min_score,
    )
    out_path = args.out or Path(f"eval/results_{args.chunk_size}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved evaluation report to {out_path}")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
