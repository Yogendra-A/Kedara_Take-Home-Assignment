"""Evaluation runner for a single FAISS index configuration."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

from config.settings import LLMConfig, RetrievalConfig, get_index_dir
from eval.metrics import summarize_metrics
from src.generator import LLMClient, generate_grounded_answer
from src.retriever import Embedder, SentenceTransformerEmbedder, VectorStore

REQUIRED_QUESTION_FIELDS = {
    "question_id",
    "question_text",
    "answerability",
    "question_category",
    "expected_source_doc_ids",
    "expected_answer_facts",
}


def load_eval_questions(questions_path: Path) -> List[Dict[str, Any]]:
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
        if item["answerability"] not in {"answerable", "unanswerable"}:
            raise ValueError(
                f"Question {qid} has invalid answerability: {item['answerability']!r}"
            )
    return items


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
        result = generate_grounded_answer(
            question=q["question_text"],
            retrieved_chunks=retrieved,
            client=llm_client,
            config=llm_config,
            min_top_score=min_score,
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
                "answer": result.answer,
                "citations": result.citations,
                "refused": result.refused,
                "citation_valid": result.citation_valid,
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
    args = parser.parse_args()

    questions = load_eval_questions(args.questions)
    if not questions:
        print(
            f"No questions found in {args.questions}. "
            "Populate it manually using eval/questions.template.json before running evaluation.",
            file=sys.stderr,
        )
        return 1

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
