"""Grounded answer generation with strict citation validation and refusal handling."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import TYPE_CHECKING, List, Protocol

from config.settings import LLMConfig, REFUSAL_TEXT

if TYPE_CHECKING:
    from src.retriever import RetrievedChunk

CITATION_PATTERN = re.compile(r"\[([a-zA-Z0-9_-]+__c\d+__\d+)\]")

SYSTEM_PROMPT = f"""You are a careful assistant answering questions about Indian senior citizen welfare schemes, laws, healthcare, pensions, and banking.
Rules:
1. Answer ONLY using the provided context chunks. Do NOT use outside knowledge or invent facts.
2. Every factual statement must include an inline citation using the exact chunk ID in square brackets, e.g. [doc_id__c256__0000].
3. If the provided context does not contain enough information to answer the question, respond with the exact string:
{REFUSAL_TEXT}
4. Do not cite any chunk ID that is not present in the provided context."""


class LLMClient(Protocol):
    """Minimal protocol for text generation clients (supports testing without network)."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        ...


class GeneratorError(RuntimeError):
    """Raised when the LLM provider is misconfigured, rate-limited, or fails."""


@dataclass(frozen=True)
class GroundedAnswer:
    """Result of grounded answer generation, including preserved retrieved evidence."""

    question: str
    answer: str
    citations: List[str]
    retrieved_chunks: List[RetrievedChunk]
    provider: str
    model: str
    refused: bool
    citation_valid: bool


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


def format_context(retrieved_chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks with stable chunk IDs and provenance for the prompt."""
    blocks: List[str] = []
    for rc in retrieved_chunks:
        c = rc.chunk
        blocks.append(
            f"[{c.chunk_id}] (doc_id={c.doc_id}, title={c.title}, section={c.section}, "
            f"score={rc.score:.4f}, url={c.source_url})\n{c.text}"
        )
    return "\n\n".join(blocks)


def extract_and_validate_citations(
    answer_text: str, retrieved_chunks: List[RetrievedChunk]
) -> tuple[List[str], bool]:
    """Extract [chunk_id] citations and verify every cited ID belongs to retrieved_chunks."""
    allowed_ids = {rc.chunk.chunk_id for rc in retrieved_chunks}
    found = CITATION_PATTERN.findall(answer_text)
    unique_found = list(dict.fromkeys(found))
    if not unique_found:
        return [], False
    all_valid = all(cid in allowed_ids for cid in unique_found)
    return unique_found, all_valid


def generate_grounded_answer(
    question: str,
    retrieved_chunks: List[RetrievedChunk],
    client: LLMClient | None = None,
    config: LLMConfig | None = None,
    min_top_score: float = 0.0,
) -> GroundedAnswer:
    """Generate a grounded answer with validated chunk-level citations or exact refusal."""
    cfg = config or LLMConfig()
    if not question.strip() or not retrieved_chunks:
        return GroundedAnswer(
            question=question,
            answer=REFUSAL_TEXT,
            citations=[],
            retrieved_chunks=retrieved_chunks,
            provider=cfg.provider,
            model=cfg.groq_model if cfg.provider == "groq" else cfg.gemini_model,
            refused=True,
            citation_valid=True,
        )

    if min_top_score > 0.0 and retrieved_chunks[0].score < min_top_score:
        return GroundedAnswer(
            question=question,
            answer=REFUSAL_TEXT,
            citations=[],
            retrieved_chunks=retrieved_chunks,
            provider=cfg.provider,
            model=cfg.groq_model if cfg.provider == "groq" else cfg.gemini_model,
            refused=True,
            citation_valid=True,
        )

    if client is None:
        client, provider, model = build_llm_client(cfg)
    else:
        provider = cfg.provider
        model = getattr(client, "model", "custom")

    context_str = format_context(retrieved_chunks)
    user_prompt = f"Context:\n{context_str}\n\nQuestion: {question}\n\nAnswer:"
    raw_answer = client.generate(SYSTEM_PROMPT, user_prompt).strip()

    if raw_answer == REFUSAL_TEXT or REFUSAL_TEXT in raw_answer and len(raw_answer) <= len(REFUSAL_TEXT) + 5:
        return GroundedAnswer(
            question=question,
            answer=REFUSAL_TEXT,
            citations=[],
            retrieved_chunks=retrieved_chunks,
            provider=provider,
            model=model,
            refused=True,
            citation_valid=True,
        )

    citations, valid = extract_and_validate_citations(raw_answer, retrieved_chunks)
    if not valid:
        return GroundedAnswer(
            question=question,
            answer=REFUSAL_TEXT,
            citations=citations,
            retrieved_chunks=retrieved_chunks,
            provider=provider,
            model=model,
            refused=True,
            citation_valid=False,
        )

    return GroundedAnswer(
        question=question,
        answer=raw_answer,
        citations=citations,
        retrieved_chunks=retrieved_chunks,
        provider=provider,
        model=model,
        refused=False,
        citation_valid=True,
    )
