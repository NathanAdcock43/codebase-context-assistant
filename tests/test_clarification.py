from __future__ import annotations

"""
ROLE: Verify deterministic clarification suggestions for weak fuzzy retrieval.
LAYER: tests
FLOW: retrieval_clarification_validation

INPUTS:
- fuzzy developer questions
- unrelated developer questions

OUTPUTS:
- clarification suggestion assertions
- no-suggestion assertions for unrelated questions

UPSTREAM:
- clarification suggestion module

DOWNSTREAM:
- local test runs
- future CI guardrails
- ask workflow clarification behavior
- API and CLI clarification responses

OWNS:
- deterministic clarification rule tests
- related-term suggestion tests
- fuzzy HTTP API question tests

DOES_NOT_OWN:
- retrieval scoring tests
- ask workflow orchestration tests
- API route tests
- CLI output tests
- LLM query rewriting tests

SIDE_EFFECTS:
- none

STATE:
  reads:
    - none
  writes:
    - none

NOTES:
- These tests should remain deterministic.
- Suggestions are search aids, not source evidence.
"""

from code_context.clarification import (
    should_suggest_clarification_for_weak_alignment,
    suggest_clarification,
)
from code_context.models import SearchResult, SourceChunk


def test_suggest_clarification_for_fuzzy_http_api_question() -> None:
    suggestion = suggest_clarification("What part lets another program talk to this?")

    assert suggestion is not None
    assert suggestion.suggested_question == (
        "Where is the HTTP API layer implemented, and which routes does it expose?"
    )
    assert suggestion.suggested_related_terms == [
        "HTTP",
        "API",
        "FastAPI",
        "endpoint",
        "route",
    ]
    assert "HTTP API" in suggestion.reason



def test_should_suggest_clarification_for_docs_only_fuzzy_implementation_question() -> None:
    source = SearchResult(
        chunk=_chunk(
            relative_path="docs/project_brief.md",
            content="The project exposes a FastAPI API for codebase questions.",
        ),
        score=0.8,
    )

    assert should_suggest_clarification_for_weak_alignment(
        question="What part lets another program talk to this?",
        sources=[source],
        related_terms=[],
    ) is True


def test_should_not_suggest_clarification_when_implementation_source_is_retrieved() -> None:
    source = SearchResult(
        chunk=_chunk(
            relative_path="src/code_context/api.py",
            content="def create_app():\n    app = FastAPI()",
        ),
        score=0.8,
    )

    assert should_suggest_clarification_for_weak_alignment(
        question="What part lets another program talk to this?",
        sources=[source],
        related_terms=[],
    ) is False


def test_should_not_suggest_clarification_when_user_supplied_related_terms() -> None:
    source = SearchResult(
        chunk=_chunk(
            relative_path="docs/project_brief.md",
            content="The project exposes a FastAPI API for codebase questions.",
        ),
        score=0.8,
    )

    assert should_suggest_clarification_for_weak_alignment(
        question="What part lets another program talk to this?",
        sources=[source],
        related_terms=["HTTP", "API"],
    ) is False


def _chunk(*, relative_path: str, content: str) -> SourceChunk:
    return SourceChunk(
        chunk_id=f"{relative_path}:1-1",
        relative_path=relative_path,
        start_line=1,
        end_line=1,
        content=content,
        language="python",
    )



def test_suggest_clarification_returns_none_for_unmatched_question() -> None:
    assert suggest_clarification("Where is the payroll export formatter?") is None
