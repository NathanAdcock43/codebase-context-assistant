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
- adjacent same-file source chunks
- long source blocks with structural code markers

OUTPUTS:
- LlmRequest construction assertions
- grounded source formatting assertions
- prompt validation assertions
- source truncation assertions
- source head/tail truncation assertions
- adjacent same-file source merge assertions
- structural outline assertions

UPSTREAM:
- prompt builder module
- grounded answer generation service
- LLM client boundary
- SearchResult model
- SourceChunk model

DOWNSTREAM:
- local test runs
- future CI guardrails
- OpenAI provider adapter tests
- grounded answer generation tests
- future responder node LLM integration tests
- system map review

OWNS:
- grounded answer prompt tests
- generated answer source-use rule tests
- adjacent same-file source merge tests
- structural outline tests
- source context formatting tests
- prompt validation tests
- source truncation tests
- source head/tail truncation tests

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
    build_source_outline,
    format_search_result_context,
    merge_adjacent_results_for_prompt,
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



def test_grounded_answer_system_prompt_rejects_speculation_and_hypothetical_code() -> None:
    assert "Treat the supplied source chunks as the complete evidence available" in (
        DEFAULT_GROUNDED_ANSWER_SYSTEM_PROMPT
    )
    assert "Do not say that a file, endpoint, function, or behavior likely exists." in (
        DEFAULT_GROUNDED_ANSWER_SYSTEM_PROMPT
    )
    assert "Do not include hypothetical code blocks or reconstructed implementations" in (
        DEFAULT_GROUNDED_ANSWER_SYSTEM_PROMPT
    )
    assert "list only items visible in the supplied sources" in DEFAULT_GROUNDED_ANSWER_SYSTEM_PROMPT

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
    assert "Do not infer hidden endpoints, files, functions, or behavior from naming patterns." in prompt
    assert "Do not use speculative phrases such as likely, probably, or similar" in prompt
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



def test_build_grounded_answer_user_prompt_merges_adjacent_same_file_chunks() -> None:
    prompt = build_grounded_answer_user_prompt(
        question="Where are API endpoints defined?",
        results=[
            _result(
                relative_path="src/code_context/api.py",
                start_line=1,
                end_line=3,
                content="line 1\nline 2\nline 3",
            ),
            _result(
                relative_path="src/code_context/api.py",
                start_line=3,
                end_line=5,
                content="line 3\nline 4\nline 5",
            ),
            _result(
                relative_path="docs/project_brief.md",
                start_line=1,
                end_line=2,
                content="brief 1\nbrief 2",
                language="markdown",
            ),
        ],
        max_source_chars_per_result=80,
    )

    assert "Source 1:" in prompt
    assert "File: src/code_context/api.py" in prompt
    assert "Lines: 1-5" in prompt
    assert prompt.count("line 3") == 1
    assert "line 4" in prompt
    assert "line 5" in prompt
    assert "Source 2:" in prompt
    assert "File: docs/project_brief.md" in prompt
    assert "Source 3:" not in prompt


def test_merge_adjacent_results_for_prompt_does_not_merge_different_files() -> None:
    merged_results = merge_adjacent_results_for_prompt(
        [
            _result(relative_path="first.py", start_line=1, end_line=2, content="first"),
            _result(relative_path="second.py", start_line=3, end_line=4, content="second"),
        ]
    )

    assert [result.chunk.relative_path for result in merged_results] == ["first.py", "second.py"]



def test_build_source_outline_includes_python_classes_functions_and_decorators() -> None:
    content = "\n".join(
        [
            "class ApiController:",
            "    pass",
            "",
            '@app.get("/health")',
            "def health():",
            '    return HealthResponse(status="ok")',
            "",
            "async def load_status():",
            '    return "ok"',
        ]
    )

    outline = build_source_outline(
        content=content,
        start_line=10,
        language="python",
    )

    assert "- line 10: class ApiController:" in outline
    assert '- line 13: @app.get("/health")' in outline
    assert "- line 14: def health():" in outline
    assert "- line 17: async def load_status():" in outline


def test_format_search_result_context_includes_structural_outline_when_content_is_truncated() -> None:
    content = "\n".join(
        [
            "module header",
            *["filler"] * 20,
            '@app.get("/health")',
            "def health():",
            '    return HealthResponse(status="ok")',
            *["filler"] * 80,
        ]
    )

    context = format_search_result_context(
        _result(
            relative_path="src/code_context/api.py",
            start_line=1,
            end_line=104,
            content=content,
            language="python",
        ),
        source_number=1,
        max_source_chars=120,
    )

    assert "Structural outline:" in context
    assert '- line 22: @app.get("/health")' in context
    assert "- line 23: def health():" in context
    assert "... [truncated middle]" in context


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
    assert "... [truncated middle]" in truncated



def test_truncate_source_content_preserves_head_and_tail_context() -> None:
    content = "HEADER_CONTEXT:" + (" middle " * 40) + "ROUTE_DECORATOR_CONTEXT"

    truncated = truncate_source_content(content, max_chars=90)

    assert len(truncated) <= 90
    assert truncated.startswith("HEADER_CONTEXT")
    assert "ROUTE_DECORATOR_CONTEXT" in truncated
    assert "... [truncated middle]" in truncated


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