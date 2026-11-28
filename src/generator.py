"""Grounded answer generation with short source handles (S1..Sn), citation resolution, and failure categorization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol

from config.settings import LLMConfig, REFUSAL_TEXT

if TYPE_CHECKING:
    from src.retriever import RetrievedChunk

HANDLE_PATTERN = re.compile(r"[\[\u3010](S\d+)[\]\u3011]")

SYSTEM_PROMPT = f"""You are a careful assistant answering questions about Indian senior citizen welfare schemes, laws, healthcare, pensions, and banking.
Rules:
1. Answer ONLY using the provided numbered sources. Do NOT use outside knowledge, extrapolate, or invent facts.
2. Verify exact entity and scope match: only use numbers, rates, limits, or rules that explicitly apply to the exact scheme or subject asked about. Never transfer figures from a different scheme mentioned in the same passage.
3. Verify temporal and factual completeness: if the question asks for "current" values and the sources only provide dated/historical figures, an indirect ceiling, or omit a required part of the question, do NOT guess or generalize.
4. Cite every factual statement inline using the exact source handle in square brackets, such as [S1] or [S2], using ONLY the handles provided in the context.
5. If the provided sources do not directly and unambiguously contain enough information to answer the question, respond with the exact string:
{REFUSAL_TEXT}
6. Do not cite any handle that is not in the provided context."""


class LLMClient(Protocol):
    """Minimal protocol for text generation clients (supports testing without network)."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...


class GeneratorError(RuntimeError):
    """Raised when the LLM provider is misconfigured, rate-limited, or fails."""


@dataclass(frozen=True)
class GroundedAnswer:
    """Result of grounded answer generation with resolved provenance and failure category."""

    question: str
    answer: str
    citations: List[str]
    retrieved_chunks: List[RetrievedChunk]
    provider: str
    model: str
    refused: bool
    citation_valid: bool
    error_type: str = "none"
    initial_failure: Optional[str] = None
    cited_handles: List[str] = None  # type: ignore[assignment]
    citation_provenance: List[Dict[str, Any]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.cited_handles is None:
            object.__setattr__(self, "cited_handles", [])
        if self.citation_provenance is None:
            object.__setattr__(self, "citation_provenance", [])


class GroqClient:
    """Thin wrapper around the official Groq Python SDK."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.0, max_tokens: int = 512) -> None:
        if not api_key or api_key == "your_groq_api_key_here":
            raise GeneratorError("Missing GROQ_API_KEY. Set it in .env or environment variables.")
        from groq import Groq

        self._client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            msg = str(exc).lower()
            if "rate" in msg or "429" in msg:
                raise GeneratorError(f"Groq rate limit exceeded: {exc}") from exc
            raise GeneratorError(f"Groq API error: {exc}") from exc


class GeminiClient:
    """Thin wrapper around the official google-genai SDK."""

    def __init__(self, api_key: str, model: str, temperature: float = 0.0, max_tokens: int = 512) -> None:
        if not api_key or api_key == "your_gemini_api_key_here":
            raise GeneratorError("Missing GEMINI_API_KEY. Set it in .env or environment variables.")
        from google import genai
        from google.genai import types

        self._client = genai.Client(api_key=api_key)
        self._types = types
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=user_prompt,
                config=self._types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    temperature=self.temperature,
                    max_output_tokens=self.max_tokens,
                ),
            )
            return (resp.text or "").strip()
        except Exception as exc:
            msg = str(exc).lower()
            if "429" in msg or "quota" in msg or "rate" in msg:
                raise GeneratorError(f"Gemini rate limit exceeded: {exc}") from exc
            raise GeneratorError(f"Gemini API error: {exc}") from exc


def build_llm_client(config: LLMConfig | None = None) -> tuple[LLMClient, str, str]:
    """Instantiate the configured LLMClient (groq or gemini) and return (client, provider, model)."""
    cfg = config or LLMConfig()
    provider = cfg.provider.strip().lower()
    if provider == "groq":
        return (
            GroqClient(
                api_key=cfg.groq_api_key,
                model=cfg.groq_model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            ),
            "groq",
            cfg.groq_model,
        )
    if provider == "gemini":
        return (
            GeminiClient(
                api_key=cfg.gemini_api_key,
                model=cfg.gemini_model,
                temperature=cfg.temperature,
                max_tokens=cfg.max_tokens,
            ),
            "gemini",
            cfg.gemini_model,
        )
    raise GeneratorError(f"Unsupported LLM_PROVIDER: {cfg.provider}")


def build_handle_map(retrieved_chunks: List[RetrievedChunk]) -> Dict[str, RetrievedChunk]:
    """Deterministically map S1..Sn handles to retrieved chunks in rank order."""
    return {f"S{i}": rc for i, rc in enumerate(retrieved_chunks, start=1)}


def format_context(retrieved_chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks with deterministic short source handles [S1]..[Sn]."""
    handle_map = build_handle_map(retrieved_chunks)
    blocks: List[str] = []
    for handle, rc in handle_map.items():
        c = rc.chunk
        blocks.append(
            f"[{handle}] (title={c.title}, section={c.section}, doc_id={c.doc_id}, "
            f"score={rc.score:.4f}, url={c.source_url})\n{c.text}"
        )
    return "\n\n".join(blocks)


def resolve_handle_citations(
    answer_text: str, handle_map: Dict[str, RetrievedChunk]
) -> tuple[List[str], List[str], List[Dict[str, Any]], bool]:
    """Extract [S1]..[Sn] handles, validate membership, and resolve to chunk_ids and provenance."""
    found_handles = list(dict.fromkeys(HANDLE_PATTERN.findall(answer_text)))
    if not found_handles:
        return [], [], [], False

    if not all(h in handle_map for h in found_handles):
        return found_handles, [], [], False

    resolved_chunk_ids: List[str] = []
    provenance: List[Dict[str, Any]] = []
    for h in found_handles:
        rc = handle_map[h]
        c = rc.chunk
        resolved_chunk_ids.append(c.chunk_id)
        provenance.append(
            {
                "handle": h,
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "title": c.title,
                "section": c.section,
                "source_url": c.source_url,
                "score": rc.score,
            }
        )
    return found_handles, resolved_chunk_ids, provenance, True


def _is_refusal_text(text: str) -> bool:
    cleaned = text.strip().strip('"').strip("'")
    return cleaned == REFUSAL_TEXT or (
        REFUSAL_TEXT in cleaned and len(cleaned) <= len(REFUSAL_TEXT) + 10
    )


_ENTITY_STOP_WORDS = {
    "what", "when", "which", "who", "whom", "whose", "where", "why", "how",
    "does", "do", "did", "are", "is", "was", "were", "can", "could", "would",
    "should", "will", "if", "my", "under", "in", "on", "for", "of", "to",
    "the", "a", "an", "india", "indian", "and", "or", "both", "all",
}
_CONNECTOR_WORDS = {"of", "for", "and", "to", "in", "on", "the"}
_ENTITY_PHRASE_RE = re.compile(
    r"\b[A-Z][A-Za-z0-9-]*(?:\s+(?:of|for|and|to|in|on|the|[A-Z][A-Za-z0-9-]*))*\b"
)
_RECENCY_RE = re.compile(
    r"\b(current|currently|present|presently|latest|today|now|as of now)\b",
    re.IGNORECASE,
)
_PERIOD_BOUND_YEAR_RE = re.compile(
    r"\b(?:for the year|upto|up to|until|as on|during|full month|effective from|valid till)\b"
    r"[^.\n]{0,40}?\b(20\d{2})(?:-(\d{2}))?\b",
    re.IGNORECASE,
)


def _normalize_token(token: str) -> str:
    cleaned = re.sub(r"[^a-z0-9]", "", token.lower())
    if len(cleaned) > 3 and cleaned.endswith("s") and not cleaned.endswith("ss"):
        cleaned = cleaned[:-1]
    return cleaned


def _extract_question_entities(question: str) -> List[str]:
    """Extract capitalized entity/scheme phrases and acronyms from a question."""
    entities: List[str] = []
    for phrase in _ENTITY_PHRASE_RE.findall(question):
        words = phrase.split()
        while words and words[0].lower() in _ENTITY_STOP_WORDS:
            words.pop(0)
        while words and words[-1].lower() in _CONNECTOR_WORDS:
            words.pop()
        if not words:
            continue
        cleaned = " ".join(words)
        if cleaned.lower() in _ENTITY_STOP_WORDS or len(cleaned) < 2:
            continue
        entities.append(cleaned)
    return list(dict.fromkeys(entities))


def _entity_in_text(entity: str, text: str) -> bool:
    """Check whether an entity (or its singularized tokens / acronym) appears in text."""
    if not entity or not text:
        return False
    norm_e = re.sub(r"[^a-z0-9]+", " ", entity.lower()).strip()
    norm_t = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    if norm_e and norm_e in norm_t:
        return True

    e_tokens = [
        _normalize_token(w) for w in entity.split() if w.lower() not in _CONNECTOR_WORDS
    ]
    if not e_tokens:
        return False
    t_tokens = {_normalize_token(w) for w in text.split()}
    t_tokens.update(_normalize_token(w) for w in re.split(r"[\s\-/(),.:;]+", text) if w)
    if all(et in t_tokens for et in e_tokens):
        return True

    if len(e_tokens) >= 2:
        acro = "".join(
            w[0].lower()
            for w in entity.split()
            if w[0].isalpha() and w.lower() not in _CONNECTOR_WORDS
        )
        if len(acro) >= 3 and re.search(rf"\b{re.escape(acro)}\b", text.lower()):
            return True
    return False


def _normalize_number(num_str: str) -> str:
    cleaned = num_str.replace(",", "").strip()
    if cleaned.endswith(".0"):
        cleaned = cleaned[:-2]
    return cleaned


def _extract_numeric_claims(answer_text: str) -> List[tuple[str, str]]:
    """Extract normalized percentage and monetary/scaled numeric claims from an answer."""
    cleaned = HANDLE_PATTERN.sub("", answer_text)
    cleaned = re.sub(r"(?m)^\s*\d+\.\s+", "", cleaned)
    claims: List[tuple[str, str]] = []

    for m in re.finditer(r"(\d+(?:,\d+)*(?:\.\d+)?)\s*(?:%|per\s*cent|percent\b)", cleaned, re.I):
        claims.append(("percent", _normalize_number(m.group(1))))

    for m in re.finditer(
        r"(?:₹|Rs\.?|INR\.?)\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*(lakh|crore|thousand|million)?",
        cleaned,
        re.IGNORECASE,
    ):
        scale = (m.group(2) or "").lower()
        claims.append(("amount", f"{_normalize_number(m.group(1))}{(' ' + scale) if scale else ''}"))

    for m in re.finditer(r"\b(\d+(?:,\d+)*(?:\.\d+)?)\s+(lakh|crore)\b", cleaned, re.IGNORECASE):
        claims.append(("amount", f"{_normalize_number(m.group(1))} {m.group(2).lower()}"))

    return list(dict.fromkeys(claims))


def _split_sentences(text: str) -> List[str]:
    """Split text into sentences while preserving abbreviations like Rs., INR., No., i.e."""
    protected = re.sub(r"\b(Rs|INR|No|Dr|Mr|Mrs|Ms|St|i\.e|e\.g)\.", r"\1<DOT>", text, flags=re.I)
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", protected) if s.strip()]
    return [p.replace("<DOT>", ".") for p in parts]


def _find_supporting_sentences(kind: str, val: str, text: str) -> List[str]:
    """Return sentences in text that contain the given normalized numeric claim."""
    matched: List[str] = []
    for sent in _split_sentences(text):
        s_norm = re.sub(r"(\d),(\d)", r"\1\2", sent)
        s_norm = re.sub(r"\s+%", "%", s_norm)
        if kind == "percent":
            if re.search(rf"(?<!\d){re.escape(val)}(?:%|\s*per\s*cent\b|\s*percent\b)", s_norm, re.I):
                matched.append(sent)
        elif kind == "amount":
            parts = val.split()
            num = parts[0]
            scale = parts[1] if len(parts) > 1 else ""
            if scale:
                if re.search(rf"(?<!\d){re.escape(num)}\s*{re.escape(scale)}", s_norm, re.I):
                    matched.append(sent)
            elif re.search(rf"(?:₹|Rs\.?|INR\.?)\s*{re.escape(num)}(?!\d)", s_norm, re.I):
                matched.append(sent)
    return matched


def verify_answer_evidence(
    question: str,
    answer_text: str,
    cited_chunks: List[RetrievedChunk],
    current_year: int | None = None,
) -> tuple[bool, str]:
    """Verify entity matching, numeric claim attribution, and temporal alignment against cited chunks."""
    if not cited_chunks:
        return False, "no_cited_chunks"

    entities = _extract_question_entities(question)
    for ent in entities:
        if not any(
            _entity_in_text(ent, f"{rc.chunk.title} {rc.chunk.section} {rc.chunk.text}")
            for rc in cited_chunks
        ):
            return False, f"entity_mismatch:{ent}"

    claims = _extract_numeric_claims(answer_text)
    for kind, val in claims:
        supporting: List[tuple[RetrievedChunk, str]] = []
        for rc in cited_chunks:
            for sent in _find_supporting_sentences(kind, val, rc.chunk.text):
                supporting.append((rc, sent))
        if not supporting:
            return False, f"unsupported_numeric:{kind}={val}"

        if entities:
            attributed = False
            for rc, sent in supporting:
                header = f"{rc.chunk.title} {rc.chunk.section}"
                if all(_entity_in_text(ent, header) for ent in entities):
                    attributed = True
                    break
                if all(_entity_in_text(ent, sent) for ent in entities):
                    if not re.search(r"\b(ceiling|threshold)\s+of\b", sent, re.IGNORECASE):
                        attributed = True
                        break
            if not attributed:
                return False, f"numeric_entity_mismatch:{kind}={val}"

    all_text = " ".join(rc.chunk.text for rc in cited_chunks)
    target_years = re.findall(r"\b(20\d{2})\b", question)
    for yr in target_years:
        if yr not in all_text:
            return False, f"temporal_year_missing:{yr}"

    if _RECENCY_RE.search(question):
        ref_year = current_year if current_year is not None else datetime.now().year
        bounded_years: List[int] = []
        for m in _PERIOD_BOUND_YEAR_RE.finditer(all_text):
            yr = int(m.group(1))
            if m.group(2):
                yr = 2000 + int(m.group(2))
            bounded_years.append(yr)
        recent_years = [
            int(y) for y in re.findall(r"\b(20\d{2})\b", all_text) if int(y) >= ref_year - 1
        ]
        if bounded_years and max(bounded_years) < ref_year - 1 and not recent_years:
            return False, f"stale_temporal_evidence:{max(bounded_years)}"

    return True, "ok"


def generate_grounded_answer(
    question: str,
    retrieved_chunks: List[RetrievedChunk],
    client: LLMClient | None = None,
    config: LLMConfig | None = None,
    min_top_score: float = 0.0,
) -> GroundedAnswer:
    """Generate a grounded answer with S1..Sn citation resolution, single format retry, and failure separation."""
    cfg = config or LLMConfig()
    default_model = cfg.groq_model if cfg.provider == "groq" else cfg.gemini_model

    if not question.strip() or not retrieved_chunks:
        return GroundedAnswer(
            question=question,
            answer=REFUSAL_TEXT,
            citations=[],
            retrieved_chunks=retrieved_chunks,
            provider=cfg.provider,
            model=default_model,
            refused=True,
            citation_valid=True,
            error_type="evidence_insufficient",
        )

    if min_top_score > 0.0 and retrieved_chunks[0].score < min_top_score:
        return GroundedAnswer(
            question=question,
            answer=REFUSAL_TEXT,
            citations=[],
            retrieved_chunks=retrieved_chunks,
            provider=cfg.provider,
            model=default_model,
            refused=True,
            citation_valid=True,
            error_type="evidence_insufficient",
        )

    if client is None:
        client, provider, model = build_llm_client(cfg)
    else:
        provider = cfg.provider
        model = getattr(client, "model", "custom")

    handle_map = build_handle_map(retrieved_chunks)
    valid_handles_str = ", ".join(f"[{h}]" for h in handle_map)
    context_str = format_context(retrieved_chunks)
    user_prompt = (
        f"Available source handles: {valid_handles_str}\n\n"
        f"Context:\n{context_str}\n\n"
        f"Question: {question}\n\nAnswer:"
    )

    raw_answer = client.generate(SYSTEM_PROMPT, user_prompt).strip()
    if _is_refusal_text(raw_answer):
        return GroundedAnswer(
            question=question,
            answer=REFUSAL_TEXT,
            citations=[],
            retrieved_chunks=retrieved_chunks,
            provider=provider,
            model=model,
            refused=True,
            citation_valid=True,
            error_type="evidence_insufficient",
        )

    handles, chunk_ids, provenance, valid = resolve_handle_citations(raw_answer, handle_map)
    initial_failure: Optional[str] = None

    if not valid:
        initial_failure = "malformed_citations"
        retry_prompt = (
            f"{user_prompt}\n{raw_answer}\n\n"
            f"Your previous response had missing or unknown citations. "
            f"Rewrite the answer citing ONLY valid handles from ({valid_handles_str}) in square brackets, "
            f"or respond with exact text: {REFUSAL_TEXT}"
        )
        retry_answer = client.generate(SYSTEM_PROMPT, retry_prompt).strip()
        if _is_refusal_text(retry_answer):
            return GroundedAnswer(
                question=question,
                answer=REFUSAL_TEXT,
                citations=[],
                retrieved_chunks=retrieved_chunks,
                provider=provider,
                model=model,
                refused=True,
                citation_valid=True,
                error_type="evidence_insufficient",
                initial_failure=initial_failure,
            )
        handles, chunk_ids, provenance, valid = resolve_handle_citations(retry_answer, handle_map)
        if not valid:
            return GroundedAnswer(
                question=question,
                answer=retry_answer,
                citations=[],
                retrieved_chunks=retrieved_chunks,
                provider=provider,
                model=model,
                refused=False,
                citation_valid=False,
                error_type="malformed_citations",
                initial_failure=initial_failure,
                cited_handles=handles,
            )
        raw_answer = retry_answer

    cited_chunks = [handle_map[h] for h in handles]
    evidence_ok, evidence_reason = verify_answer_evidence(question, raw_answer, cited_chunks)
    if not evidence_ok:
        return GroundedAnswer(
            question=question,
            answer=REFUSAL_TEXT,
            citations=[],
            retrieved_chunks=retrieved_chunks,
            provider=provider,
            model=model,
            refused=True,
            citation_valid=True,
            error_type="evidence_insufficient",
            initial_failure=evidence_reason,
        )

    return GroundedAnswer(
        question=question,
        answer=raw_answer,
        citations=chunk_ids,
        retrieved_chunks=retrieved_chunks,
        provider=provider,
        model=model,
        refused=False,
        citation_valid=True,
        error_type="none",
        initial_failure=initial_failure,
        cited_handles=handles,
        citation_provenance=provenance,
    )
