"""Build and persist dual FAISS indexes (256-token and 512-token) from the corpus snapshot."""

import argparse
from pathlib import Path
from typing import Dict, List

from config.settings import INDEX_DIR, PROCESSED_DIR, RetrievalConfig, get_index_dir
from src.chunker import Chunk, chunk_sections
from src.corpus import load_corpus_snapshot
from src.ingest import DocumentSection
from src.retriever import Embedder, SentenceTransformerEmbedder, VectorStore


def build_dual_indexes(
    sections: List[DocumentSection] | None = None,
    processed_dir: Path = PROCESSED_DIR,
    index_base_dir: Path | None = None,
    chunk_sizes: tuple[int, ...] = (256, 512),
    overlap_tokens: int | None = None,
    embedder: Embedder | None = None,
) -> Dict[int, VectorStore]:
    """Chunk sections deterministically for each size and persist separate FAISS indexes."""
    cfg = RetrievalConfig()
    overlap = cfg.chunk_overlap_tokens if overlap_tokens is None else overlap_tokens
    if sections is None:
        sections = load_corpus_snapshot(processed_dir)
    if embedder is None:
        embedder = SentenceTransformerEmbedder(model_name=cfg.embedding_model)

    stores: Dict[int, VectorStore] = {}
    for size in chunk_sizes:
        chunks: List[Chunk] = chunk_sections(
            sections,
            chunk_size_tokens=size,
            chunk_overlap_tokens=overlap,
            tokenizer_name=cfg.tokenizer_name,
        )
        store = VectorStore(embedder=embedder)
        store.build(chunks)
        target_dir = (
            index_base_dir / f"chunks_{size}"
            if index_base_dir is not None
            else get_index_dir(size)
        )
        store.save(target_dir)
        stores[size] = store

    return stores


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build 256-token and 512-token FAISS indexes from data/processed/sections.jsonl."
    )
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    args = parser.parse_args()

    stores = build_dual_indexes(
        processed_dir=args.processed_dir,
        index_base_dir=args.index_dir,
    )
    for size, store in sorted(stores.items()):
        print(
            f"Built chunks_{size} index in {args.index_dir / f'chunks_{size}'}: "
            f"{store.index.ntotal} vectors"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
