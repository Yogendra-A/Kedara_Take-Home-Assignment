"""Controlled comparison between 256-token and 512-token FAISS indexes."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

from config.settings import LLMConfig, RetrievalConfig
from eval.run_eval import evaluate_index, load_eval_questions
from src.generator import build_llm_client
from src.retriever import SentenceTransformerEmbedder


def select_worst_examples(outcomes: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    """Select up to `limit` worst-performing outcomes for qualitative inspection."""

    def _severity(o: Dict[str, Any]) -> tuple[int, float]:
        ans = o.get("answerability")
        refused = bool(o.get("refused"))
        err = o.get("error_type", "none")
        retrieved = o.get("retrieved_doc_ids", [])
        expected = set(o.get("expected_source_doc_ids", []))
        top_score = float((o.get("retrieved_scores") or [0.0])[0])

        if err == "provider_error":
            rank = 0
        elif not refused and o.get("citation_valid") is False:
            rank = 1
        elif ans == "unanswerable" and not refused:
            rank = 2
        elif ans == "answerable" and refused:
            rank = 3
        elif ans == "answerable" and expected and not any(d in expected for d in retrieved):
            rank = 4
        else:
            rank = 5
        return (rank, top_score)

    ranked = sorted(outcomes, key=_severity)
    return ranked[:limit]


def _fmt_metric(val: Any) -> str:
    if val is None:
        return "Pending"
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def format_comparison_markdown(comparison: Dict[str, Any]) -> str:
    """Render a concise Markdown evaluation report from comparison results."""
    s256 = comparison.get("summary_256", {})
    s512 = comparison.get("summary_512", {})
    settings = comparison.get("settings", {})

    rows = [
        ("Answerable Questions", "answerable_count"),
        ("Unanswerable Questions", "unanswerable_count"),
        ("Recall@5 (Answerable)", "recall_at_5"),
        ("MRR@5 (Answerable)", "mrr_at_5"),
        ("Refusal Recall (Unanswerable)", "refusal_recall"),
        ("False-Refusal Rate (Answerable)", "false_refusal_rate"),
        ("Citation Valid Rate", "citation_valid_rate"),
        ("Missing Citations Count", "missing_citations_count"),
        ("Malformed Citations Count", "malformed_citations_count"),
        ("Provider Error Count", "provider_error_count"),
        ("Human-Reviewed Citation Correctness", "citation_correctness_human_reviewed"),
    ]

    lines = [
        "# Kedara — Evaluation Report (256 vs 512 Tokens)",
        "",
        f"- **Embedding Model**: `{settings.get('embedding_model', '')}`",
        f"- **Top-K**: `{settings.get('top_k', 5)}` | **Min Score**: `{settings.get('min_score', 0.2)}`",
        f"- **LLM**: `{settings.get('llm_provider', '')}` (`{settings.get('llm_model', '')}`, `temperature={settings.get('temperature', 0.0)}`)",
        "",
        "## 256 vs 512 Comparison Table",
        "",
        "| Metric | 256-token Index | 512-token Index |",
        "| --- | --- | --- |",
    ]
    for label, key in rows:
        lines.append(f"| {label} | {_fmt_metric(s256.get(key))} | {_fmt_metric(s512.get(key))} |")

    lines.extend(["", "## Five Worst-Answer Examples (256-token Index)", ""])
    worst = select_worst_examples(comparison.get("results_256", {}).get("outcomes", []), limit=5)
    for idx, item in enumerate(worst, start=1):
        lines.append(
            f"{idx}. **{item.get('question_id')}** ({item.get('answerability')}) — "
            f"`error_type={item.get('error_type')}`, `refused={item.get('refused')}`, "
            f"`citation_valid={item.get('citation_valid')}`"
        )
        lines.append(f"   - Question: {item.get('question_text')}")
        lines.append(f"   - Expected Docs: `{item.get('expected_source_doc_ids')}` | Retrieved Docs: `{item.get('retrieved_doc_ids')}`")
        lines.append(f"   - Answer: {item.get('answer')}")
    lines.append("")
    return "\n".join(lines)


def run_comparison(
    questions_path: Path = Path("eval/questions.json"),
    top_k: int = 5,
    min_score: float = 0.2,
    out_path: Path = Path("eval/comparison_256_vs_512.json"),
) -> Dict[str, Any]:
    """Evaluate both 256-token and 512-token indexes with identical settings."""
    questions = load_eval_questions(questions_path)
    if not questions:
        raise ValueError(
            f"No questions found in {questions_path}. "
            "Add manually written questions first."
        )

    cfg = RetrievalConfig()
    llm_cfg = LLMConfig()
    embedder = SentenceTransformerEmbedder(model_name=cfg.embedding_model)
    llm_client, _, _ = build_llm_client(llm_cfg)

    res_256 = evaluate_index(
        questions=questions,
        chunk_size=256,
        top_k=top_k,
        min_score=min_score,
        embedder=embedder,
        llm_client=llm_client,
        llm_config=llm_cfg,
    )
    res_512 = evaluate_index(
        questions=questions,
        chunk_size=512,
        top_k=top_k,
        min_score=min_score,
        embedder=embedder,
        llm_client=llm_client,
        llm_config=llm_cfg,
    )

    comparison = {
        "settings": {
            "embedding_model": cfg.embedding_model,
            "top_k": top_k,
            "min_score": min_score,
            "llm_provider": llm_cfg.provider,
            "llm_model": llm_cfg.groq_model if llm_cfg.provider == "groq" else llm_cfg.gemini_model,
            "temperature": llm_cfg.temperature,
            "max_tokens": llm_cfg.max_tokens,
        },
        "summary_256": res_256["summary"],
        "summary_512": res_512["summary"],
        "results_256": res_256,
        "results_512": res_512,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path = out_path.with_name("EVALUATION_REPORT.md")
    report_path.write_text(format_comparison_markdown(comparison), encoding="utf-8")
    return comparison


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare 256-token vs 512-token RAG indexes.")
    parser.add_argument("--questions", type=Path, default=Path("eval/questions.json"))
    parser.add_argument("--top-k", type=int, default=RetrievalConfig().top_k)
    parser.add_argument("--min-score", type=float, default=0.2)
    parser.add_argument("--out", type=Path, default=Path("eval/comparison_256_vs_512.json"))
    args = parser.parse_args()

    try:
        comparison = run_comparison(
            questions_path=args.questions,
            top_k=args.top_k,
            min_score=args.min_score,
            out_path=args.out,
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Saved controlled comparison to {args.out}")
    print("\n=== 256-token Summary ===")
    print(json.dumps(comparison["summary_256"], indent=2))
    print("\n=== 512-token Summary ===")
    print(json.dumps(comparison["summary_512"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
