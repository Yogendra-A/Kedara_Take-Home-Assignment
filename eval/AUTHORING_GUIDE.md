# Evaluation Question Authoring Guide

This guide explains how to manually author the 25-question evaluation set in `eval/questions.json` without fabricating questions or ground-truth labels.

Use `eval/questions.template.json` as the schema reference. Each entry must be a JSON object with:
- `question_id`: Unique string (e.g., `"Q01"` .. `"Q25"`)
- `question_text`: Non-empty question string
- `answerability`: `"answerable"` or `"unanswerable"`
- `question_category`: Short category label (e.g., `"helpline_support"`, `"pension"`, `"healthcare"`, `"out_of_corpus"`)
- `expected_source_doc_ids`: List of `doc_id`s for `"answerable"` questions; `[]` for `"unanswerable"` questions
- `expected_answer_facts`: List of concise factual statements expected in a correct answer for `"answerable"` questions; `[]` for `"unanswerable"` questions

---

## 1. Inspect Extracted Government Documents

All 24 successfully extracted documents are stored in `data/processed/sections.jsonl` (with per-document extraction counts in `data/processed/ingestion_report.json`).

List all available `doc_id`s, titles, section counts, and character counts:

```powershell
python -c "import json; from pathlib import Path; r = json.loads(Path('data/processed/ingestion_report.json').read_text(encoding='utf-8')); [print(f\"{d['doc_id']:30s} | ok={d['extraction_ok']} | secs={d['section_count']:2d} | chars={d['char_count']:6d} | {d['title']}\") for d in r['records']]"
```

Read the extracted text of a specific `doc_id` (for example, `elderline_14567`):

```powershell
python -c "import json; from pathlib import Path; secs = [json.loads(l) for l in Path('data/processed/sections.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]; [print(f'=== {s[\"section\"]} ===\n{s[\"text\"]}\n') for s in secs if s['doc_id'] == 'elderline_14567']"
```

## 2. Find Matching Document IDs

Search across all extracted sections for a keyword (e.g., `"14567"`, `"Ayushman"`, `"doorstep"`, `"80TTB"`) to see which `doc_id`s actually contain the answer:

```powershell
python -c "import json; from pathlib import Path; kw = 'doorstep'.lower(); secs = [json.loads(l) for l in Path('data/processed/sections.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]; [print(f\"{s['doc_id']} | {s['section']} | {s['text'][:160]}...\") for s in secs if kw in s['text'].lower()]"
```

Only assign `doc_id`s to `expected_source_doc_ids` after verifying that the extracted text in `data/processed/sections.jsonl` actually contains the answer. Note that `pib_vayoshreshtha_samman` failed extraction (`0` chars) and is not in the index.

## 3. Write Expected Answer Facts

For each `"answerable"` question:
- Write 1 to 4 short, verifiable factual statements in `expected_answer_facts` taken directly from the inspected `sections.jsonl` text (e.g., `["Toll-free helpline number is 14567", "Operational from 8 AM to 8 PM"]`).
- Avoid vague phrasing; include specific numbers, eligibility thresholds, or scheme names present in the source text.

## 4. Identify Genuinely Unanswerable Questions

Include unanswerable questions (e.g., 5–7 out of 25) whose answers are **not** present anywhere in `data/processed/sections.jsonl`. Good categories:
- State-specific schemes or municipal benefits not covered in the national corpus
- Non-Indian programs (e.g., US Medicare, UK State Pension)
- Facts from the failed PDF (`pib_vayoshreshtha_samman`) or details absent from short snippets (verify via keyword search in step 2 that no other document mentions them)
- Current-year budgetary figures or personal account-specific queries

Set `"answerability": "unanswerable"`, `"expected_source_doc_ids": []`, and `"expected_answer_facts": []`.

## 5. Validate Evaluation JSON

Validate schema, uniqueness, answerability rules, and `doc_id` existence against `data/processed/sections.jsonl` without calling any LLM:

```powershell
python -m eval.run_eval --validate-only --questions eval/questions.json
```

(Or in Docker: `docker run --rm -v "${PWD}:/app" kedara-rag:py311 python -m eval.run_eval --validate-only`)

## 6. Run Both Configurations

Ensure both FAISS indexes are built (`python -m src.index_builder`), then run either single-index evaluation or the controlled 256 vs 512 comparison:

```powershell
# Single configuration
python -m eval.run_eval --chunk-size 256 --questions eval/questions.json
python -m eval.run_eval --chunk-size 512 --questions eval/questions.json

# Controlled comparison (identical questions, embedder, top_k, min_score, and LLM config)
python -m eval.compare --questions eval/questions.json
```

Outputs are saved to `eval/results_256.json`, `eval/results_512.json`, and `eval/comparison_256_vs_512.json` with `human_citation_correct: null` ready for manual citation review.
