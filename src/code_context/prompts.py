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
- adjacent same-file retrieval chunks for generated answer context

OUTPUTS:
- provider-neutral LlmRequest records
- system prompt text
- user prompt text with source references
- user prompt source blocks with adjacent same-file chunks merged for continuity

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
- adjacent same-file source chunk merging for prompt context
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
- Adjacent chunks from the same file should be merged for generated-answer prompt context when possible.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from code_context.llm import LlmMessage, LlmRequest
from code_context.models import SearchResult, SourceChunk


@dataclass(frozen=True)
class _PromptSource:
    result: SearchResult
    merged_chunk_count: int


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

    prompt_sources = _merge_adjacent_results_for_prompt(selected_results)

    source_blocks = [
        format_search_result_context(
            prompt_source.result,
            source_number=index,
            max_source_chars=max_source_chars_per_result * prompt_source.merged_chunk_count,
        )
        for index, prompt_source in enumerate(prompt_sources, start=1)
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


def merge_adjacent_results_for_prompt(results: Sequence[SearchResult]) -> list[SearchResult]:
    """Merge adjacent same-file retrieval results into continuous prompt context blocks."""

    return [prompt_source.result for prompt_source in _merge_adjacent_results_for_prompt(results)]


def truncate_source_content(content: str, *, max_chars: int) -> str:
    """Truncate source content for prompt safety while preserving a clear marker."""

    _validate_source_limit(max_chars)

    if len(content) <= max_chars:
        return content

    marker = "\n... [truncated]"
    if max_chars <= len(marker):
        return marker[-max_chars:]

    return f"{content[: max_chars - len(marker)].rstrip()}{marker}"


def _merge_adjacent_results_for_prompt(results: Sequence[SearchResult]) -> list[_PromptSource]:
    if not results:
        return []

    prompt_sources: list[_PromptSource] = []

    current_result = results[0]
    current_count = 1

    for next_result in results[1:]:
        if _can_merge_results(current_result, next_result):
            current_result = _merge_search_results(current_result, next_result)
            current_count += 1
            continue

        prompt_sources.append(_PromptSource(result=current_result, merged_chunk_count=current_count))
        current_result = next_result
        current_count = 1

    prompt_sources.append(_PromptSource(result=current_result, merged_chunk_count=current_count))

    return prompt_sources


def _can_merge_results(left: SearchResult, right: SearchResult) -> bool:
    left_chunk = left.chunk
    right_chunk = right.chunk

    return (
        left_chunk.relative_path == right_chunk.relative_path
        and left_chunk.language == right_chunk.language
        and right_chunk.start_line >= left_chunk.start_line
        and right_chunk.start_line <= left_chunk.end_line + 1
    )


def _merge_search_results(left: SearchResult, right: SearchResult) -> SearchResult:
    left_chunk = left.chunk
    right_chunk = right.chunk
    merged_start_line = left_chunk.start_line
    merged_end_line = max(left_chunk.end_line, right_chunk.end_line)
    merged_content = _merge_chunk_content(
        left_content=left_chunk.content,
        left_end_line=left_chunk.end_line,
        right_content=right_chunk.content,
        right_start_line=right_chunk.start_line,
    )

    return SearchResult(
        chunk=SourceChunk(
            chunk_id=f"{left_chunk.relative_path}:{merged_start_line}-{merged_end_line}:merged",
            relative_path=left_chunk.relative_path,
            start_line=merged_start_line,
            end_line=merged_end_line,
            content=merged_content,
            language=left_chunk.language,
        ),
        score=max(left.score, right.score),
    )


def _merge_chunk_content(
    *,
    left_content: str,
    left_end_line: int,
    right_content: str,
    right_start_line: int,
) -> str:
    left_lines = left_content.splitlines()
    right_lines = right_content.splitlines()
    overlapping_line_count = max(0, left_end_line - right_start_line + 1)

    if overlapping_line_count >= len(right_lines):
        return "\n".join(left_lines)

    merged_lines = [*left_lines, *right_lines[overlapping_line_count:]]
    return "\n".join(merged_lines)


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