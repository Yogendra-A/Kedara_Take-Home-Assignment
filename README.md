# Kedara — Ask-a-Care-Resource RAG Pipeline

A minimal, grounded Retrieval-Augmented Generation (RAG) system for answering questions about official Indian senior-citizen welfare schemes, statutory rights, healthcare, pensions, and banking services.

## Architecture

- **`config/settings.py`**: Central paths, retrieval defaults (`all-MiniLM-L6-v2`, `256`/`512`-token chunks, `32`-token overlap, `top_k=5`), and LLM provider configuration (`groq` / `gemini`).
- **`src/ingest.py` & `src/corpus.py`**: Manifest validation (`data/manifest.json`), HTTP fetching with exponential backoff, HTML/PDF section extraction, boilerplate stripping, section deduplication, and reproducible snapshot/report persistence (`data/processed/`).
- **`src/chunker.py`**: Deterministic `tiktoken` (`cl100k_base`) chunking with overlap and unique chunk IDs (`{doc_id}__c{size}__{idx:04d}`).
- **`src/retriever.py` & `src/index_builder.py`**: L2-normalized `sentence-transformers` embeddings indexed in FAISS (`IndexFlatL2`) with cosine similarity scoring and dual-index building (`chunks_256` and `chunks_512`).
- **`src/generator.py` & `src/ask.py`**: Grounded answering with citation validation, strict refusal (`"I don't know from these sources"`), and CLI interface.
- **`eval/`**: Schema-validated evaluation runner (`eval/run_eval.py`), Recall@5 / MRR@5 / refusal / human-reviewed citation metrics (`eval/metrics.py`), and controlled `256` vs `512` comparison (`eval/compare.py`).

## Running with Docker (Python 3.11)

```powershell
docker build --network=host -t kedara-rag:py311 .
docker run --rm -v "${PWD}:/app" kedara-rag:py311 python -m pytest -v
```

### Asking a Question

```powershell
docker run --rm --network=host --env-file .env -v "${PWD}:/app" kedara-rag:py311 `
  python -m src.ask "What is the Elderline helpline number for senior citizens?"
```

### Running Evaluation

1. Populate `eval/questions.json` using `eval/questions.template.json`.
2. Run single-index evaluation or controlled comparison:

```powershell
docker run --rm --network=host --env-file .env -v "${PWD}:/app" kedara-rag:py311 `
  python -m eval.run_eval --chunk-size 256

docker run --rm --network=host --env-file .env -v "${PWD}:/app" kedara-rag:py311 `
  python -m eval.compare
```

## Current Status & Incomplete Features

- **Completed**: 25-document official Indian corpus manifest (`22` documents extracted), dual FAISS indexes (`740` vectors at 256 tokens; `385` vectors at 512 tokens; `384`-D), unit/integration tests (`25/25` passing in Docker), and evaluation scaffolding.
- **In Progress / Incomplete**:
  - Citation handle simplification and false-refusal reduction (Checkpoint 3.5).
  - Manual evaluation question set (`eval/questions.json` is intentionally empty awaiting human-authored questions and human citation review).
  - No frontend/UI (CLI only).
