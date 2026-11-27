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

## Pre-Fix Grounding Defect & Error Analysis (Manual Testing)

- **Question**: `"What is the current annual interest rate and maximum deposit limit for the Senior Citizens Savings Scheme (SCSS) in India?"`
- **Original Failed Output (Preserved)**:
  > `"The Senior Citizens Savings Scheme (SCSS) currently offers an assured rate of return of up to 7.75% per annum, and the maximum deposit limit is ₹15 lakh per senior citizen."`
- **Original Citation**: `[S1]` (`pib_mwpsc_act_implementation__c256__0019`, `doc_id=pib_mwpsc_act_implementation`, `section=PROBLEMS OF AGED PEOPLE`, `similarity=0.6027`).
- **Root Cause**:
  - **Retrieval**: `scss_pib` was excluded (navigation-only), so no SCSS document exists in the corpus. Dense retrieval returned `pib_mwpsc_act_implementation__c256__0019` because it mentions SCSS in passing while describing a *different* scheme (LIC's PMVVY: 7.40% for 2020–21, annual reset in line with SCSS up to a 7.75% ceiling, ₹15 lakh maximum purchase price).
  - **Generation**: Without explicit entity-scope and temporal-validity rules in `SYSTEM_PROMPT`, the LLM transferred PMVVY's 2020–23 ceiling (7.75%) and purchase limit (₹15 lakh) onto SCSS and presented them as current facts.
- **Fix & Verification**: Tightened `SYSTEM_PROMPT` in `src/generator.py` with entity/scope-match and temporal/completeness refusal rules (no hardcoded scheme rules, no `min_score` threshold increase). Re-running the SCSS query returned `"I don't know from these sources"` (`Refused: True`), while Elderline (`14567`) and AB PM-JAY (`₹5 lakh per family per year`) continued to answer with valid citations.
