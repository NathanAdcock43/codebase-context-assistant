from __future__ import annotations

"""
ROLE: Build grounded LLM prompts from verified retrieval results.
LAYER: prompts
FLOW: grounded_answer_prompting

INPUTS:
- developer question text
- grounded SearchResult records
- LLM model name
- prompt temperature setting
- result count limits
- source content character limits

OUTPUTS:
- provider-neutral LlmRequest records
- system prompt text
- user prompt text with source references

UPSTREAM:
- grounded answer generation service
- verified retrieval results from the retrieval and verifier path
- future responder node LLM integration
- local tests

DOWNSTREAM:
- LLM client boundary
- OpenAI provider adapter
- generated answer synthesis
- API and CLI generated ask responses

OWNS:
- grounded answer system prompt text
- grounded answer user prompt construction
- generated answer source-use rules
- source context formatting
- source content truncation
- prompt input validation

DOES_NOT_OWN:
- LLM provider calls
- retrieval scoring
- retrieval sufficiency decisions
- stale-index refusal
- citation verification after generation
- API routing
- CLI argument parsing
- LangGraph workflow routing

SIDE_EFFECTS:
- none

STATE:
  reads:
    - in-memory SearchResult records
  writes:
    - none

NOTES:
- This module does not call an LLM.
- This prompt builder should only be used after stale-index and sufficiency checks have passed.
- The prompt tells the configured LLM to answer only from supplied sources and to refuse if the sources are insufficient.
- The prompt should discourage speculative wording such as likely, probably, or inferred endpoint behavior.
"""

from collections.abc import Sequence

from code_context.llm import LlmMessage, LlmRequest
from code_context.models import SearchResult


DEFAULT_GROUNDED_ANSWER_SYSTEM_PROMPT = """
You are a codebase context assistant.

Answer the developer question using only the supplied source context.

Rules:
- Use only the provided sources.
- Treat the supplied source chunks as the complete evidence available for this answer.
- Do not invent files, functions, endpoints, behavior, architecture, or relationships.
- Do not say that a file, endpoint, function, or behavior likely exists.
- Do not include hypothetical code blocks or reconstructed implementations unless the exact code appears in the supplied sources.
- When listing endpoints, functions, classes, or files, list only items visible in the supplied sources.
- If documentation mentions a capability but the implementation is not visible in the supplied sources, say that directly.
- Cite relevant file paths and line ranges in the answer.
- If the supplied sources are not enough, say that the indexed context is insufficient.
- Keep the answer practical and implementation-focused.
""".strip()


def build_grounded_answer_request(
    *,
    question: str,
    results: Sequence[SearchResult],
    model: str,
    temperature: float = 0.2,
    max_results: int = 5,
    max_source_chars_per_result: int = 1200,
) -> LlmRequest:
    """Build a provider-neutral request for grounded answer generation."""

    user_prompt = build_grounded_answer_user_prompt(
        question=question,
        results=results,
        max_results=max_results,
        max_source_chars_per_result=max_source_chars_per_result,
    )

    return LlmRequest(
        messages=[
            LlmMessage(role="system", content=DEFAULT_GROUNDED_ANSWER_SYSTEM_PROMPT),
            LlmMessage(role="user", content=user_prompt),
        ],
        model=model,
        temperature=temperature,
    )


def build_grounded_answer_user_prompt(
    *,
    question: str,
    results: Sequence[SearchResult],
    max_results: int = 5,
    max_source_chars_per_result: int = 1200,
) -> str:
    """Build the user prompt containing the question and grounded sources."""

    normalized_question = _normalize_question(question)
    selected_results = _select_results(results, max_results=max_results)
    _validate_source_limit(max_source_chars_per_result)

    source_blocks = [
        format_search_result_context(result, source_number=index, max_source_chars=max_source_chars_per_result)
        for index, result in enumerate(selected_results, start=1)
    ]

    return "\n\n".join(
        [
            "Developer question:",
            normalized_question,
            "Grounded source context:",
            "\n\n".join(source_blocks),
            "Answer requirements:",
            "- Answer only from the grounded source context above.",
            "- Include file paths and line ranges when referencing implementation details.",
            "- Do not infer hidden endpoints, files, functions, or behavior from naming patterns.",
            "- Do not use speculative phrases such as likely, probably, or similar when describing implementation details.",
            "- If documentation mentions a capability but implementation code is not visible in the supplied context, say that directly.",
            "- If these sources are insufficient, say so directly instead of guessing.",
        ]
    )


def format_search_result_context(result: SearchResult, *, source_number: int, max_source_chars: int = 1200) -> str:
    """Format a retrieved source chunk for prompt context."""

    if source_number < 1:
        msg = "Source number must be at least 1."
        raise ValueError(msg)

    _validate_source_limit(max_source_chars)

    chunk = result.chunk
    truncated_content = truncate_source_content(chunk.content, max_chars=max_source_chars)

    return "\n".join(
        [
            f"Source {source_number}:",
            f"File: {chunk.relative_path}",
            f"Lines: {chunk.start_line}-{chunk.end_line}",
            f"Language: {chunk.language}",
            f"Retrieval score: {result.score:.4f}",
            "Content:",
            truncated_content,
        ]
    )


def truncate_source_content(content: str, *, max_chars: int) -> str:
    """Truncate source content for prompt safety while preserving a clear marker."""

    _validate_source_limit(max_chars)

    if len(content) <= max_chars:
        return content

    marker = "\n... [truncated]"
    if max_chars <= len(marker):
        return marker[-max_chars:]

    return f"{content[: max_chars - len(marker)].rstrip()}{marker}"


def _normalize_question(question: str) -> str:
    normalized = question.strip()
    if not normalized:
        msg = "Question cannot be empty."
        raise ValueError(msg)
    return normalized


def _select_results(results: Sequence[SearchResult], *, max_results: int) -> list[SearchResult]:
    if max_results < 1:
        msg = "Maximum result count must be at least 1."
        raise ValueError(msg)

    if not results:
        msg = "At least one grounded search result is required."
        raise ValueError(msg)

    return list(results[:max_results])


def _validate_source_limit(max_source_chars: int) -> None:
    if max_source_chars < 50:
        msg = "Maximum source characters must be at least 50."
        raise ValueError(msg)