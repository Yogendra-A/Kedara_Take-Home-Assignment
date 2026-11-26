"""Minimal Streamlit UI for the Kedara Care Resource Explorer."""

from __future__ import annotations

from dataclasses import dataclass
import re
import time
from typing import Any, Dict, List, Optional

import streamlit as st

from config.settings import LLMConfig, REFUSAL_TEXT, RetrievalConfig, get_index_dir
from src.generator import GeneratorError, GroundedAnswer, LLMClient, generate_grounded_answer
from src.retriever import RetrievedChunk, SentenceTransformerEmbedder, VectorStore

SAMPLE_QUESTIONS: List[tuple[str, str]] = [
    ("Elderline Helpline", "What is the Elderline helpline number for senior citizens?"),
    ("RBI Doorstep Banking", "What doorstep banking facilities does RBI mandate for senior citizens?"),
    ("Out-of-Corpus Check", "Does Ayushman Bharat cover dental implants in the United States?"),
]

_KEY_REDACTION_PATTERNS = [
    re.compile(r"(gsk_[A-Za-z0-9_-]{8,})"),
    re.compile(r"(AIza[A-Za-z0-9_-]{12,})"),
    re.compile(r"(sk-[A-Za-z0-9_-]{10,})"),
    re.compile(r"(Bearer\s+[A-Za-z0-9._-]{8,})", re.IGNORECASE),
]


@dataclass(frozen=True)
class QueryOutcome:
    """Structured UI result returned by execute_rag_query."""

    question: str
    retrieved_chunks: List[RetrievedChunk]
    grounded_answer: Optional[GroundedAnswer]
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    error_message: Optional[str] = None


def sanitize_error_message(message: str) -> str:
    """Redact potential API keys or bearer tokens before displaying errors in the UI."""
    cleaned = message
    for pattern in _KEY_REDACTION_PATTERNS:
        cleaned = pattern.sub("[REDACTED]", cleaned)
    return cleaned


def format_latency(ms: float) -> str:
    """Format a millisecond duration with explicit units."""
    if ms < 0:
        ms = 0.0
    if ms < 1000.0:
        return f"{ms:.1f} ms"
    return f"{ms / 1000.0:.2f} s ({ms:.0f} ms)"


def get_corpus_doc_count(store: VectorStore) -> int:
    """Return the number of unique documents present in the loaded VectorStore."""
    if store.is_empty:
        return 0
    return len({chunk.doc_id for chunk in store.chunks})


def build_citation_rows(result: GroundedAnswer) -> List[Dict[str, Any]]:
    """Extract structured citation provenance rows for display."""
    rows: List[Dict[str, Any]] = []
    for item in result.citation_provenance:
        rows.append(
            {
                "handle": str(item.get("handle", "")),
                "title": str(item.get("title", "")),
                "section": str(item.get("section", "")),
                "doc_id": str(item.get("doc_id", "")),
                "chunk_id": str(item.get("chunk_id", "")),
                "source_url": str(item.get("source_url", "")),
                "score": float(item.get("score", 0.0)),
            }
        )
    return rows


def build_evidence_rows(retrieved_chunks: List[RetrievedChunk]) -> List[Dict[str, Any]]:
    """Transform retrieved chunks into structured evidence items."""
    rows: List[Dict[str, Any]] = []
    for rank, rc in enumerate(retrieved_chunks, start=1):
        c = rc.chunk
        rows.append(
            {
                "rank": rank,
                "score": float(rc.score),
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "title": c.title,
                "section": c.section,
                "source_url": c.source_url,
                "text": c.text,
            }
        )
    return rows


def execute_rag_query(
    question: str,
    store: VectorStore,
    top_k: int = 5,
    min_score: float = 0.2,
    llm_client: LLMClient | None = None,
    llm_config: LLMConfig | None = None,
) -> QueryOutcome:
    """Run retrieval and grounded generation with perf_counter latency measurement."""
    cleaned_q = question.strip()
    if not cleaned_q:
        return QueryOutcome(
            question=question,
            retrieved_chunks=[],
            grounded_answer=None,
            retrieval_ms=0.0,
            generation_ms=0.0,
            total_ms=0.0,
            error_message="Please enter a question to search.",
        )

    if store.is_empty:
        return QueryOutcome(
            question=cleaned_q,
            retrieved_chunks=[],
            grounded_answer=None,
            retrieval_ms=0.0,
            generation_ms=0.0,
            total_ms=0.0,
            error_message=(
                "The FAISS index is missing or empty. Run `python -m src.index_builder` "
                "to build both indexes from `data/processed/sections.jsonl`."
            ),
        )

    t0 = time.perf_counter()
    retrieved = store.search(cleaned_q, top_k=top_k)
    t1 = time.perf_counter()
    retrieval_ms = (t1 - t0) * 1000.0

    t2 = time.perf_counter()
    try:
        grounded = generate_grounded_answer(
            question=cleaned_q,
            retrieved_chunks=retrieved,
            client=llm_client,
            config=llm_config,
            min_top_score=min_score,
        )
        t3 = time.perf_counter()
        generation_ms = (t3 - t2) * 1000.0
        return QueryOutcome(
            question=cleaned_q,
            retrieved_chunks=retrieved,
            grounded_answer=grounded,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=retrieval_ms + generation_ms,
            error_message=None,
        )
    except GeneratorError as exc:
        t3 = time.perf_counter()
        generation_ms = (t3 - t2) * 1000.0
        return QueryOutcome(
            question=cleaned_q,
            retrieved_chunks=retrieved,
            grounded_answer=None,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=retrieval_ms + generation_ms,
            error_message=sanitize_error_message(str(exc)),
        )


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder(model_name: str) -> SentenceTransformerEmbedder:
    """Cache the SentenceTransformer embedder keyed by model name."""
    return SentenceTransformerEmbedder(model_name=model_name)


@st.cache_resource(show_spinner="Loading FAISS index...")
def load_vector_store(chunk_size: int, model_name: str) -> VectorStore:
    """Cache the FAISS VectorStore keyed by chunk_size and embedding model."""
    embedder = load_embedder(model_name)
    return VectorStore.load(get_index_dir(chunk_size), embedder=embedder)


def _apply_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #F7F8FA;
            color: #1E293B;
        }
        .block-container {
            max-width: 980px;
            padding-top: 2.2rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3, .explorer-title {
            color: #1B365D !important;
            font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            letter-spacing: -0.01em;
        }
        .explorer-subtitle {
            color: #475569;
            font-size: 1.05rem;
            margin-top: -0.4rem;
            margin-bottom: 0.75rem;
        }
        .corpus-badge {
            display: inline-block;
            background-color: #E6F4F1;
            color: #0F766E;
            border: 1px solid #B2DFDB;
            border-radius: 6px;
            padding: 0.25rem 0.65rem;
            font-size: 0.85rem;
            font-weight: 600;
            margin-bottom: 1.25rem;
        }
        .panel-card {
            background-color: #FFFFFF;
            border: 1px solid #DCE3EA;
            border-radius: 8px;
            padding: 1.25rem 1.4rem;
            margin-top: 0.75rem;
            margin-bottom: 1rem;
        }
        .latency-bar {
            color: #475569;
            font-size: 0.86rem;
            padding: 0.45rem 0.75rem;
            background-color: #F1F5F9;
            border-left: 3px solid #0F766E;
            border-radius: 4px;
            margin: 0.6rem 0 1.1rem 0;
        }
        .citation-card {
            background-color: #FFFFFF;
            border: 1px solid #DCE3EA;
            border-left: 3px solid #0F766E;
            border-radius: 6px;
            padding: 0.75rem 1rem;
            margin-bottom: 0.55rem;
        }
        div.stButton > button[kind="primary"] {
            background-color: #0F766E;
            border-color: #0F766E;
            color: #FFFFFF;
        }
        div.stButton > button[kind="primary"]:hover {
            background-color: #0D5E58;
            border-color: #0D5E58;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Care Resource Explorer",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _apply_styles()

    cfg = RetrievalConfig()

    st.markdown("<h1 class='explorer-title'>Care Resource Explorer</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p class='explorer-subtitle'>Explore trusted information on Indian senior-citizen schemes and services.</p>",
        unsafe_allow_html=True,
    )

    if "question_input" not in st.session_state:
        st.session_state["question_input"] = ""
    if "trigger_search" not in st.session_state:
        st.session_state["trigger_search"] = False

    def _select_sample(sample_q: str) -> None:
        st.session_state["question_input"] = sample_q
        st.session_state["trigger_search"] = True

    col_cfg, col_spacer = st.columns([2, 3])
    with col_cfg:
        chunk_size = st.radio(
            "Index chunk configuration",
            options=[256, 512],
            format_func=lambda s: f"{s}-token chunks",
            horizontal=True,
        )

    store_error: Optional[str] = None
    store = VectorStore(index=None, chunks=[])
    try:
        store = load_vector_store(chunk_size=chunk_size, model_name=cfg.embedding_model)
    except Exception as exc:
        store_error = sanitize_error_message(str(exc))

    doc_count = get_corpus_doc_count(store)
    chunk_count = len(store.chunks)
    st.markdown(
        f"<div class='corpus-badge'>{doc_count} official documents indexed · "
        f"{chunk_count} chunks ({chunk_size}-token index)</div>",
        unsafe_allow_html=True,
    )

    if store_error:
        st.error(f"Index loading error: {store_error}")

    with st.form("search_form", clear_on_submit=False):
        question = st.text_input(
            "Question",
            key="question_input",
            placeholder="Ask about Indian senior-citizen welfare schemes, Elderline 14567, pensions, healthcare, or RBI banking...",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Search", type="primary")

    st.caption("Sample queries:")
    sample_cols = st.columns(len(SAMPLE_QUESTIONS))
    for idx, (label, sample_q) in enumerate(SAMPLE_QUESTIONS):
        sample_cols[idx].button(
            label,
            key=f"sample_btn_{idx}",
            use_container_width=True,
            on_click=_select_sample,
            args=(sample_q,),
        )

    should_run = submitted or st.session_state.pop("trigger_search", False)
    if should_run:
        active_question = st.session_state.get("question_input", question)
        with st.spinner("Retrieving passages and generating grounded answer..."):
            outcome = execute_rag_query(
                question=active_question,
                store=store,
                top_k=cfg.top_k,
                min_score=0.2,
            )
        st.session_state["last_outcome"] = outcome
        st.session_state["last_chunk_size"] = chunk_size

    outcome: Optional[QueryOutcome] = st.session_state.get("last_outcome")
    if outcome is not None:
        used_chunk_size = st.session_state.get("last_chunk_size", chunk_size)
        st.divider()
        st.subheader("Answer")

        st.markdown(
            f"<div class='latency-bar'>"
            f"<strong>Retrieval:</strong> {format_latency(outcome.retrieval_ms)} &nbsp;|&nbsp; "
            f"<strong>Generation:</strong> {format_latency(outcome.generation_ms)} &nbsp;|&nbsp; "
            f"<strong>Total:</strong> {format_latency(outcome.total_ms)} &nbsp;|&nbsp; "
            f"<strong>Index:</strong> {used_chunk_size}-token"
            f"</div>",
            unsafe_allow_html=True,
        )

        if outcome.error_message:
            st.error(outcome.error_message)
        elif outcome.grounded_answer is not None:
            res = outcome.grounded_answer
            if res.refused:
                st.warning(REFUSAL_TEXT)
                st.caption(
                    "No retrieved passage met the grounding threshold or contained enough evidence to answer."
                )
            else:
                with st.container(border=True):
                    if not res.citation_valid:
                        st.warning(
                            f"Citation validation failed ({res.error_type}). Verify against the retrieved evidence below."
                        )
                    # Render model output safely via standard Markdown (no raw HTML injection).
                    st.markdown(res.answer)
                    st.caption(f"Provider: {res.provider} ({res.model})")

                citation_rows = build_citation_rows(res)
                if citation_rows:
                    st.subheader("Supporting Citations")
                    for row in citation_rows:
                        with st.container(border=True):
                            st.markdown(
                                f"**[{row['handle']}]** [{row['title']}]({row['source_url']})"
                            )
                            st.caption(
                                f"Section: {row['section']} · Doc ID: `{row['doc_id']}` · "
                                f"Chunk ID: `{row['chunk_id']}` · Cosine similarity: `{row['score']:.4f}`"
                            )

        evidence_rows = build_evidence_rows(outcome.retrieved_chunks)
        if evidence_rows:
            st.subheader(f"Retrieved Evidence ({len(evidence_rows)} passages)")
            for ev in evidence_rows:
                header = (
                    f"{ev['rank']}. {ev['title']} — {ev['section']} "
                    f"(similarity {ev['score']:.4f})"
                )
                with st.expander(header, expanded=False):
                    st.markdown(f"**Official source:** [{ev['source_url']}]({ev['source_url']})")
                    st.caption(f"Doc ID: `{ev['doc_id']}` · Chunk ID: `{ev['chunk_id']}`")
                    st.text(ev["text"])


if __name__ == "__main__":
    main()
