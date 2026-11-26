"""Controlled comparison between 256-token and 512-token FAISS indexes."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict

from config.settings import LLMConfig, RetrievalConfig
from eval.run_eval import evaluate_index, load_eval_questions
from src.generator import build_llm_client
from src.retriever import SentenceTransformerEmbedder


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
