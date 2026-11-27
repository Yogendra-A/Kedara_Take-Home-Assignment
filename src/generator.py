"""Grounded answer generation with short source handles (S1..Sn), citation resolution, and failure categorization."""

from __future__ import annotations

from dataclasses import dataclass
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
