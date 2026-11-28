# Kedara — Evaluation Report (256 vs 512 Tokens)

> **Baseline Preservation Notice**: The 25-question metrics in this report reflect the **pre-fix baseline** stored in `eval/comparison_256_vs_512.json` (preserved unmodified). Post-fix verification results are documented separately in the **Post-Fix Regression & End-to-End Checks** section below.

- **Corpus**: `21` relevant official documents (`96` deduplicated sections; `3` noisy/irrelevant and `1` failed PDF excluded)
- **Embedding Model**: `all-MiniLM-L6-v2` (`384`-D L2-normalized)
- **Top-K**: `5` | **Min Score**: `0.2`
- **LLM**: `groq` (`openai/gpt-oss-20b`, `temperature=0.0`, `max_tokens=1024`)
- **Latency Note**: Batch evaluation JSON (`eval/comparison_256_vs_512.json`) does not store per-question timing; interactive latency is measured in `app.py` via `time.perf_counter()` (live UI observations: `55.8 ms`–`1.46 s` retrieval, `936.3 ms`–`1.15 s` generation).

## 256 vs 512 Comparison Table

| Metric | 256-token Index (`508` vectors) | 512-token Index (`280` vectors) |
| --- | --- | --- |
| Total Questions | `25` (`20` answerable, `5` unanswerable) | `25` (`20` answerable, `5` unanswerable) |
| Recall@5 (Answerable) | `0.9500` (`19/20`) | `0.9500` (`19/20`) |
| MRR@5 (Answerable) | `0.8142` | `0.8625` |
| Refusal Recall (Unanswerable) | `1.0000` (`5/5`) | `0.8000` (`4/5`) |
| False-Refusal Rate (Answerable) | `0.0500` (`1/20`) | `0.0500` (`1/20`) |
| Citation Valid Rate (Syntactic) | `0.8947` (`17/19`) | `0.9500` (`19/20`) |
| Missing Citations Count | `2` (`Q14`, `Q19`) | `1` (`Q14`) |
| Malformed Citations Count | `0` | `0` |
| Provider Error Count | `0` | `0` |
| Initial Failure Recovered Count | `2` (`Q10`, `Q17`) | `3` (`Q15`, `Q18`, `Q19`) |
| Human-Reviewed Citation Correctness | `Pending` (`null` — manual review required) | `Pending` (`null` — manual review required) |

## Five Worst Actual Responses (From `eval/comparison_256_vs_512.json`)

1. **`Q21` (512-token) — Missed Refusal / Entity-Conflation Hallucination** (`unanswerable`, `refused=False`, `citation_valid=True`, `error_type=none`)
   - **Question**: *"What is the current annual interest rate for the Senior Citizens Savings Scheme, and what is its maximum deposit limit?"*
   - **Answer**: *"The Senior Citizens Savings Scheme (SCSS) offers an annual interest rate of **7.75 %**【S1】, and the maximum amount that can be deposited by a senior citizen under the scheme is **₹15 lakh**【S1】."*
   - **Retrieved Docs**: `['pib_mwpsc_act_implementation', 'incometax_senior_citizens', 'incometax_senior_citizens', 'incometax_senior_citizens', 'incometax_senior_citizens']` (`top_score=0.4996`)
   - **Failure Type**: Missed refusal / factual hallucination (`[S1]` describes LIC's PMVVY 2020–23 ceiling, not SCSS). Demonstrates why syntactic citation validity (`True`) cannot replace human citation correctness review.
2. **`Q14` (256-token & 512-token) — Empty Answer / Missing Citations** (`answerable`, `refused=False`, `citation_valid=False`, `failure_category=missing_citations`, `initial_failure=malformed_citations`)
   - **Question**: *"What are the BPL and monthly income eligibility criteria for Rashtriya Vayoshri Yojana, which documents are required, and what special delivery facility is available for people aged 80 or above?"*
   - **Answer**: `""` (empty string on both initial generation and constrained retry)
   - **Retrieved Docs**: `['dosje_rvy', 'pib_ignoaps_rvy', 'dosje_rvy', 'dosje_rvy', 'pib_senior_welfare_2']` (`top_score=0.7353`)
   - **Failure Type**: Missing citations / empty model output despite rank-1 retrieval of `dosje_rvy`.
3. **`Q16` (256-token & 512-token) — False Refusal** (`answerable`, `refused=True`, `error_type=evidence_insufficient`)
   - **Question**: *"My elderly uncle is being neglected at home, and we're worried about his safety. Is there a free helpline that can help him, what kind of support does it provide, and when can we call?"*
   - **Answer**: `"I don't know from these sources"`
   - **Retrieved Docs**: `['elderline_14567', 'elderline_14567', 'elderline_14567', 'mwpsc_act_faq_dosje', 'elderline_14567']` (`top_score=0.5516`)
   - **Failure Type**: False refusal on an informal multi-part query even though `elderline_14567` appeared at ranks 1–3 and 5.
4. **`Q19` (256-token) — Whitespace Citation Formatting Failure (`[ S1 ]`)** (`answerable`, `refused=False`, `citation_valid=False`, `failure_category=missing_citations`, `initial_failure=malformed_citations`)
   - **Question**: *"My mother is 65, and her bank is deducting TDS from her fixed-deposit interest even though her total income is low. Is there a form she can submit to avoid eligible deductions?"*
   - **Answer**: Recommended **Form 15H**, ending with `[ S1 ]` (spaces inside brackets), which failed `HANDLE_PATTERN` regex matching (`missing_citations`). Also echoed *"below the basic exemption limit"* from the adjacent Form 15G row.
   - **Retrieved Docs**: `['incometax_senior_citizens'] * 5` (`top_score=0.5309`)
   - **Failure Type**: Formatting failure (`[ S1 ]` whitespace) + minor factual nuance.
5. **`Q20` (256-token & 512-token) — Expected Source Miss (`Recall@5` Miss)** (`answerable`, `ret_hit=False`, `refused=False`, `citation_valid=True`)
   - **Question**: *"My father recently retired and wants to stay active by doing some work. Is there a government programme that connects experienced senior citizens with companies?"*
   - **Expected**: `['dosje_avyay']` | **Retrieved (256)**: `['pib_ipsrc_grants', 'pib_nphce', 'pib_ipsrc_grants', 'pib_avyay_umbrella', 'mwpsc_act_faq_dosje']` (`top_score=0.5436`)
   - **Failure Type**: Expected-document retrieval miss (model answered SCOPE/SACRED from retrieved PIB summaries, but `dosje_avyay` did not rank in top-5).

## Pre-Fix Grounding Defect & Error Analysis (Manual Testing)

- **Question**: `"What is the current annual interest rate and maximum deposit limit for the Senior Citizens Savings Scheme (SCSS) in India?"`
- **Original Failed Output (Preserved)**:
  > `"The Senior Citizens Savings Scheme (SCSS) currently offers an assured rate of return of up to 7.75% per annum, and the maximum deposit limit is ₹15 lakh per senior citizen."`
- **Original Citation**: `[S1]` (`pib_mwpsc_act_implementation__c256__0019`, `doc_id=pib_mwpsc_act_implementation`, `section=PROBLEMS OF AGED PEOPLE`, `similarity=0.6027`).
- **Root Cause**: `scss_pib` was excluded (navigation-only), so dense retrieval returned a near-miss chunk describing LIC's PMVVY (2020–23), and the LLM transferred PMVVY's ceiling (7.75%) and purchase price (₹15 lakh) onto SCSS.
- **Fix & Outcome**: Tightened `SYSTEM_PROMPT` in `src/generator.py` with entity/scope-match and temporal/completeness rules; 256-token refused `Q21` (`Refusal Recall = 1.0`), while 512-token still failed `Q21` (`Refusal Recall = 0.8`), showing that larger chunks can still induce entity conflation when a related scheme is mentioned in context.

## Post-Fix Regression & End-to-End Checks (Current System)

Prompt-only guardrails proved non-deterministic on the 256-token index as well (returning `7.75%` and `₹15 lakh` citing `[S3]` = `pib_mwpsc_act_implementation__c256__0019`, which describes PMVVY's 2020–23 ceiling rather than SCSS). To enforce grounding deterministically, `verify_answer_evidence()` was added to `src/generator.py` after `resolve_handle_citations()`.

- **Docker Full Test Suite (`kedara-rag:py311`)**: `35 passed in 43.86s` (`tests/test_app.py`, `tests/test_chunker.py`, `tests/test_corpus.py`, `tests/test_eval.py`, `tests/test_generator.py`, `tests/test_index_builder.py`, `tests/test_ingest.py`, `tests/test_retriever.py`), including regression tests for wrong-entity citations, historical figures presented as current, unsupported numerical claims, and answerable Elderline/PM-JAY queries.
- **Fresh End-to-End FAISS + Groq Checks (256-token & 512-token)**:
  - **Elderline** (*"What is the Elderline helpline number for senior citizens?"*): Supported answer (`14567`) with valid citations on both `256` (`['S1', 'S5']`) and `512` (`['S1', 'S5']`).
  - **PM-JAY 70+** (*"What health coverage is provided under Ayushman Bharat PM-JAY for citizens aged 70 and above, and does income affect eligibility?"*): Supported answer (`₹5 lakh`, irrespective of socio-economic status) with valid citations on both `256` (`['S3', 'S4']`) and `512` (`['S1', 'S2']`).
  - **SCSS** (*"What is the current annual interest rate and maximum deposit limit for the Senior Citizens Savings Scheme in India?"*): Refused with `"I don't know from these sources"` on both `256` (`pib_mwpsc_act_implementation__c256__0019` at `[S3]`, caught by `verify_answer_evidence` as `numeric_entity_mismatch:percent=7.75`) and `512` (`pib_mwpsc_act_implementation__c512__0009` at `[S1]`).
- **Rule-Based Evidence Verification Limitations**:
  1. Entity extraction relies on capitalized phrases and acronyms in the question; all-lowercase or paraphrased scheme references may not trigger entity-attribution checks.
  2. Numerical checks verify percentages and monetary/scaled amounts (`₹`, `Rs.`, `INR`, `lakh`, `crore`), not every unscaled integer or qualitative claim.
  3. Temporal checks rely on explicit period-bounding phrases (`for the year`, `upto`, `until`, `as on`, etc.) paired with past years; undated stale passages or subtle non-numeric semantic errors require full NLI or human review.
