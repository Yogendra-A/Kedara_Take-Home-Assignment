# Kedara — Evaluation Report (Template — Pending Execution)

> **Status**: Pending manual authoring of the 25-question set in `eval/questions.json` and execution of `python -m eval.compare`. No metrics are reported yet.

- **Corpus**: `21` official documents (`96` sections)
- **Embedding Model**: `all-MiniLM-L6-v2` (`384`-D L2-normalized)
- **Top-K**: `5` | **Min Score**: `0.2`
- **LLM**: `groq` (`openai/gpt-oss-20b`, `temperature=0.0`)

## 256 vs 512 Comparison Table

| Metric | 256-token Index (`508` vectors) | 512-token Index (`280` vectors) |
| --- | --- | --- |
| Answerable Questions | _Pending_ | _Pending_ |
| Unanswerable Questions | _Pending_ | _Pending_ |
| Recall@5 (Answerable) | _Pending_ | _Pending_ |
| MRR@5 (Answerable) | _Pending_ | _Pending_ |
| Refusal Recall / Correct Refusal Rate (Unanswerable) | _Pending_ | _Pending_ |
| False-Refusal Rate (Answerable) | _Pending_ | _Pending_ |
| Citation Valid Rate | _Pending_ | _Pending_ |
| Missing Citations Count | _Pending_ | _Pending_ |
| Malformed Citations Count | _Pending_ | _Pending_ |
| Provider Error Count | _Pending_ | _Pending_ |
| Human-Reviewed Citation Correctness | _Pending_ | _Pending_ |

## Per-Question Evidence & Results

- Full per-question outputs, retrieved chunks, cosine scores, cited handles, and `human_citation_correct` fields will be persisted in:
  - `eval/comparison_256_vs_512.json`
  - `eval/results_256.json`
  - `eval/results_512.json`

## Five Worst-Answer Examples (Pending Inspection)

1. _Pending evaluation run and manual review._
2. _Pending evaluation run and manual review._
3. _Pending evaluation run and manual review._
4. _Pending evaluation run and manual review._
5. _Pending evaluation run and manual review._
