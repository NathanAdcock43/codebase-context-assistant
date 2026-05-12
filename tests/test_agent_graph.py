from __future__ import annotations

"""
ROLE: Verify deterministic agent workflow runner behavior.
LAYER: tests
FLOW: deterministic_agent_workflow_validation

INPUTS:
- developer question text
- sample IndexSnapshot records
- sample SourceChunk records
- retrieval thresholds

OUTPUTS:
- completed agent workflow state assertions

UPSTREAM:
- deterministic agent workflow runner
- agent node implementation
- agent state models
- IndexSnapshot model
- SourceChunk model

DOWNSTREAM:
- local test runs
- future CI guardrails
- future FastAPI ask endpoint tests
- future LangGraph workflow tests
- system map review

OWNS:
- workflow runner tests
- full node sequence tests
- grounded answer workflow tests
- refusal workflow tests
- workflow state completion tests

DOES_NOT_OWN:
- individual node tests
- retrieval service unit tests
- API route tests
- LangGraph integration tests
- LLM answer tests

SIDE_EFFECTS:
- none

STATE:
  reads:
    - in-memory test snapshots
  writes:
    - none

NOTES:
- These tests prove the deterministic workflow can run end to end before LangGraph is added.
- The workflow should remain explainable and grounded.
"""

import pytest
from pydantic import ValidationError

from code_context.agent import AgentStepStatus, run_code_question_workflow
from code_context.index_store import build_index_snapshot
from code_context.models import IndexSnapshot, SourceChunk


def test_run_code_question_workflow_returns_grounded_answer() -> None:
    snapshot = _snapshot(
        chunks=[
            _chunk(
                chunk_id="src/scanner.py:1-2:hash",
                relative_path="src/scanner.py",
                content="def calculate_file_hash(path):\n    return hashlib.sha256(data).hexdigest()\n",
                end_line=2,
            )
        ]
    )

    state = run_code_question_workflow(
        question="How does calculate file hash work?",
        snapshot=snapshot,
        limit=1,
        minimum_top_score=0.05,
    )

    assert state.answer is not None
    assert "I found grounded indexed context for this question." in state.answer
    assert state.citations == ["src/scanner.py:1-2"]
    assert state.verification is not None
    assert state.verification.can_answer is True
    assert state.verification.is_grounded is True
    assert [(step.name, step.status) for step in state.steps] == [
        ("planner", AgentStepStatus.completed),
        ("retriever", AgentStepStatus.completed),
        ("verifier", AgentStepStatus.completed),
        ("responder", AgentStepStatus.completed),
    ]


def test_run_code_question_workflow_refuses_when_context_is_insufficient() -> None:
    snapshot = _snapshot(
        chunks=[
            _chunk(
                chunk_id="src/api.py:1-1:api",
                relative_path="src/api.py",
                content="FastAPI route definitions only",
            )
        ]
    )

    state = run_code_question_workflow(
        question="How does postgres liquibase migration work?",
        snapshot=snapshot,
        minimum_top_score=0.95,
    )

    assert state.answer == (
        "I do not have enough grounded indexed context to answer that. "
        "Reason: The best retrieved context was below the sufficiency score threshold."
    )
    assert state.citations == []
    assert state.verification is not None
    assert state.verification.can_answer is False
    assert state.verification.is_grounded is False
    assert state.steps[3].notes == [
        "Refused because verification failed: "
        "The best retrieved context was below the sufficiency score threshold."
    ]


def test_run_code_question_workflow_refuses_when_index_has_no_chunks() -> None:
    state = run_code_question_workflow(
        question="How does retrieval work?",
        snapshot=_snapshot(chunks=[]),
    )

    assert state.answer == (
        "I do not have enough grounded indexed context to answer that. "
        "Reason: The index does not contain any chunks to search."
    )
    assert state.citations == []
    assert state.retrieval is not None
    assert state.retrieval.results == []
    assert state.verification is not None
    assert state.verification.can_answer is False


def test_run_code_question_workflow_rejects_empty_question() -> None:
    with pytest.raises(ValidationError, match="question must not be empty"):
        run_code_question_workflow(
            question="   ",
            snapshot=_snapshot(chunks=[]),
        )


def _snapshot(*, chunks: list[SourceChunk]) -> IndexSnapshot:
    return build_index_snapshot(
        repo_path="C:/example/repo",
        files=[],
        chunks=chunks,
        indexed_at=123.45,
    )


def _chunk(
    *,
    content: str,
    chunk_id: str = "src/example.py:1-1:abc123",
    relative_path: str = "src/example.py",
    end_line: int | None = None,
) -> SourceChunk:
    return SourceChunk(
        chunk_id=chunk_id,
        relative_path=relative_path,
        start_line=1,
        end_line=end_line or max(1, content.count("\n")),
        content=content,
        language="python",
    )