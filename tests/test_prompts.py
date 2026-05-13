from __future__ import annotations

"""
ROLE: Verify grounded answer prompt construction.
LAYER: tests
FLOW: grounded_answer_prompt_validation

INPUTS:
- sample developer questions
- sample SearchResult records
- sample SourceChunk records
- model names
- prompt construction limits

OUTPUTS:
- LlmRequest construction assertions
- grounded source formatting assertions
- prompt validation assertions
- source truncation assertions

UPSTREAM:
- prompt builder module
- LLM client boundary
- SearchResult model
- SourceChunk model
- future LLM answer generation service

DOWNSTREAM:
- local test runs
- future CI guardrails
- future OpenAI provider tests
- future grounded answer generation tests
- future responder node LLM integration tests
- system map review

OWNS:
- grounded answer prompt tests
- source context formatting tests
- prompt validation tests
- source truncation tests

DOES_NOT_OWN:
- LLM provider client tests
- retrieval tests
- verifier tests
- stale-index refusal tests
- API route tests
- CLI tests
- LangGraph tests

SIDE_EFFECTS:
- none

STATE:
  reads:
    - in-memory test values
  writes:
    - none

NOTES:
- These tests do not call an LLM.
- The prompt builder is a contract between verified retrieval and future generated answer text.
"""

import pytest

from code_context.models import SearchResult, SourceChunk
from code_context.prompts import (
    DEFAULT_GROUNDED_ANSWER_SYSTEM_PROMPT,
    build_grounded_answer_request,
    build_grounded_answer_user_prompt,
    format_search_result_context,
    truncate_source_content,
)


def test_build_grounded_answer_request_creates_system_and_user_messages() -> None:
    request = build_grounded_answer_request(
        question="Where is scanning implemented?",
        results=[
            _result(
                relative_path="src/code_context/scanner.py",
                start_line=10,
                end_line=20,
                content="def scan_repository():\n    return []",
            )
        ],
        model="test-model",
    )

    assert request.model == "test-model"
    assert request.temperature == 0.2
    assert len(request.messages) == 2
    assert request.messages[0].role == "system"
    assert request.messages[0].content == DEFAULT_GROUNDED_ANSWER_SYSTEM_PROMPT
    assert request.messages[1].role == "user"
    assert "Where is scanning implemented?" in request.messages[1].content
    assert "src/code_context/scanner.py" in request.messages[1].content
    assert "Lines: 10-20" in request.messages[1].content


def test_build_grounded_answer_user_prompt_includes_answer_rules() -> None:
    prompt = build_grounded_answer_user_prompt(
        question="How does drift detection work?",
        results=[
            _result(
                relative_path="src/code_context/drift.py",
                start_line=1,
                end_line=5,
                content="def detect_drift():\n    return report",
            )
        ],
    )

    assert "Developer question:" in prompt
    assert "Grounded source context:" in prompt
    assert "Answer requirements:" in prompt
    assert "Answer only from the grounded source context above." in prompt
    assert "If these sources are insufficient, say so directly instead of guessing." in prompt


def test_build_grounded_answer_user_prompt_limits_result_count() -> None:
    prompt = build_grounded_answer_user_prompt(
        question="Where is configuration handled?",
        results=[
            _result(relative_path="first.py", start_line=1, end_line=2, content="first"),
            _result(relative_path="second.py", start_line=3, end_line=4, content="second"),
        ],
        max_results=1,
    )

    assert "first.py" in prompt
    assert "second.py" not in prompt


def test_format_search_result_context_includes_source_metadata_and_content() -> None:
    context = format_search_result_context(
        _result(
            relative_path="src/code_context/api.py",
            start_line=15,
            end_line=30,
            content="def create_app():\n    return app",
            language="python",
            score=0.875,
        ),
        source_number=2,
    )

    assert "Source 2:" in context
    assert "File: src/code_context/api.py" in context
    assert "Lines: 15-30" in context
    assert "Language: python" in context
    assert "Retrieval score: 0.8750" in context
    assert "def create_app()" in context


def test_truncate_source_content_adds_marker_when_content_is_too_long() -> None:
    content = "a" * 100

    truncated = truncate_source_content(content, max_chars=60)

    assert len(truncated) <= 60
    assert truncated.endswith("... [truncated]")


def test_truncate_source_content_keeps_short_content_unchanged() -> None:
    content = "short source"

    assert truncate_source_content(content, max_chars=60) == content


def test_build_grounded_answer_user_prompt_rejects_blank_question() -> None:
    with pytest.raises(ValueError, match="Question cannot be empty"):
        build_grounded_answer_user_prompt(question="   ", results=[_result()])


def test_build_grounded_answer_user_prompt_rejects_empty_results() -> None:
    with pytest.raises(ValueError, match="At least one grounded search result is required"):
        build_grounded_answer_user_prompt(question="Where is scanning?", results=[])


def test_build_grounded_answer_user_prompt_rejects_invalid_max_results() -> None:
    with pytest.raises(ValueError, match="Maximum result count must be at least 1"):
        build_grounded_answer_user_prompt(question="Where is scanning?", results=[_result()], max_results=0)


def test_format_search_result_context_rejects_invalid_source_number() -> None:
    with pytest.raises(ValueError, match="Source number must be at least 1"):
        format_search_result_context(_result(), source_number=0)


def test_truncate_source_content_rejects_too_small_limit() -> None:
    with pytest.raises(ValueError, match="Maximum source characters must be at least 50"):
        truncate_source_content("source", max_chars=49)


def _result(
    *,
    relative_path: str = "src/code_context/scanner.py",
    start_line: int = 1,
    end_line: int = 3,
    content: str = "def scan_repository():\n    return []",
    language: str = "python",
    score: float = 0.75,
) -> SearchResult:
    return SearchResult(
        chunk=SourceChunk(
            chunk_id=f"{relative_path}:{start_line}-{end_line}",
            relative_path=relative_path,
            start_line=start_line,
            end_line=end_line,
            content=content,
            language=language,
        ),
        score=score,
    )