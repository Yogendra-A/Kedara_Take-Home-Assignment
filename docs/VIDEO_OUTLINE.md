# Kedara — 4–5 Minute Video Walkthrough Outline

Target length: **4:30** (keep pacing crisp and grounded in observed behavior).

## 1. Problem & Approach (0:00 – 0:45)
- **Problem**: Indian senior citizens and caregivers face fragmented government portals (DoSJE, PIB, RBI, EPFO, Income Tax) for pensions, helplines, healthcare, and legal rights.
- **Approach**: A grounded RAG pipeline over 21 verified official Indian government documents (`96` sections) with strict `[S1]`..`[Sn]` citation provenance and explicit refusal (`"I don't know from these sources"`) when evidence is missing.

## 2. Streamlit UI Demonstration — Grounded Answer & Citations (0:45 – 1:45)
- Show **Care Resource Explorer** (`app.py`): off-white/navy/teal information-retrieval layout, dynamic corpus badge (`21 official documents indexed · 508 chunks (256-token index)`), and `256` vs `512`-token selector.
- Click **"Elderline Helpline"** (`"What is the Elderline helpline number for senior citizens?"`).
- Highlight:
  - Latency bar measured via `time.perf_counter()` (`Retrieval`, `Generation`, `Total` in `ms`/`s`).
  - Grounded answer citing `[S1]`, `[S5]` (`14567`).
  - **Supporting Citations** with clickable official `dosje.gov.in` links, section names, `doc_id`, `chunk_id`, and cosine similarity.
  - Expandable **Retrieved Evidence** excerpts.

## 3. Refusal Behavior Demonstration (1:45 – 2:25)
- Click **"Out-of-Corpus Check"** (`"Does Ayushman Bharat cover dental implants in the United States?"`).
- Show that retrieved passages (e.g., AB PM-JAY 70+ hospitalisation) do not support the claim, and the system displays the exact refusal string `"I don't know from these sources"` without hallucinating.

## 4. Architecture & Code Walkthrough (2:25 – 3:40)
- Walk through the modular pipeline:
  - `src/corpus.py` & `data/manifest.json`: 25-URL manifest, `exclude_reason` filtering for 3 noisy/unrelated pages, 1 failed scanned PDF (`pib_vayoshreshtha_samman`), 21 clean indexed documents.
  - `src/chunker.py`: deterministic `tiktoken` (`cl100k_base`) chunking at `256` (`508` vectors) vs `512` (`280` vectors) with `32`-token overlap.
  - `src/retriever.py`: `all-MiniLM-L6-v2` (`384`-D L2-normalized) + FAISS `IndexFlatL2`.
  - `src/generator.py`: short `[S1]`..`[Sn]` handles, single constrained retry on malformed citations, separate tracking of `missing_citations`, `malformed_citations`, and `provider_error`.

## 5. Experiment Setup & Honest Limitations (3:40 – 4:30)
- **Evaluation**: `eval/compare.py` runs `256` vs `512` under identical `top_k=5`, `min_score=0.2`, embedder, and LLM settings (`Recall@5`, `MRR@5`, `refusal_recall`, `false_refusal_rate`, human-reviewed citation correctness).
- **Honest Limitations**: Scanned PDF (`pib_vayoshreshtha_samman`) needs OCR; JS-heavy PIB pages (`scss_pib`, `pib_jeevan_pramaan_dlc`) yielded navigation boilerplate and were excluded; `pib_sage_portal` returned unrelated Green Railways text and was excluded; Windows Application Control blocks `_swigfaiss.pyd` on host so Docker (`kedara-rag:py311`) is used for FAISS execution.
