"""Simple CLI for retrieving chunks and generating grounded answers."""

import argparse
import sys

from config.settings import RetrievalConfig, get_index_dir
from src.generator import GeneratorError, generate_grounded_answer
from src.retriever import SentenceTransformerEmbedder, VectorStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Ask-a-Care-Resource RAG CLI")
    parser.add_argument("question", nargs="?", help="Question to ask (or omit for interactive prompt)")
    parser.add_argument("--chunk-size", type=int, default=256, choices=[256, 512], help="Index chunk size")
    parser.add_argument("--top-k", type=int, default=RetrievalConfig().top_k, help="Number of chunks to retrieve")
    parser.add_argument("--min-score", type=float, default=0.2, help="Minimum top-1 cosine similarity to attempt answer")
    parser.add_argument("--retrieve-only", action="store_true", help="Only print retrieved chunks without calling LLM")
    args = parser.parse_args()

    question = args.question or input("Question: ").strip()
    if not question:
        print("Error: Empty question.", file=sys.stderr)
        return 1

    cfg = RetrievalConfig()
    index_dir = get_index_dir(args.chunk_size)
    embedder = SentenceTransformerEmbedder(model_name=cfg.embedding_model)
    store = VectorStore.load(index_dir, embedder=embedder)
    if store.is_empty:
        print(f"Error: Index at {index_dir} is empty or missing. Build indexes first.", file=sys.stderr)
        return 1

    retrieved = store.search(question, top_k=args.top_k)
    print(f"\n=== Retrieved Chunks (chunk_size={args.chunk_size}, top_k={args.top_k}) ===")
    for i, rc in enumerate(retrieved, start=1):
        c = rc.chunk
        print(f"{i}. [{c.chunk_id}] score={rc.score:.4f} | {c.title} | section={c.section}")
        print(f"   url: {c.source_url}")
        print(f"   text: {c.text[:220]}...")

    if args.retrieve_only:
        return 0

    try:
        result = generate_grounded_answer(
            question=question,
            retrieved_chunks=retrieved,
            min_top_score=args.min_score,
        )
    except GeneratorError as exc:
        print(f"\nGenerator Error: {exc}", file=sys.stderr)
        return 2

    print("\n=== Grounded Answer ===")
    print(result.answer)
    print(f"\nCitations: {result.citations}")
    print(f"Refused: {result.refused} | Citation Valid: {result.citation_valid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
