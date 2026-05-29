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

from code_context.clarification import suggest_clarification


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


def test_suggest_clarification_returns_none_for_unmatched_question() -> None:
    assert suggest_clarification("Where is the payroll export formatter?") is None
