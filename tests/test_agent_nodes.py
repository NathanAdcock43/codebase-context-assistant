from __future__ import annotations

"""
ROLE: Verify deterministic agent node behavior.
LAYER: tests
FLOW: agent_node_validation

INPUTS:
- CodeQuestionState records
- developer question text
- existing agent steps
- sample IndexSnapshot records
- sample SourceChunk records
- sample retrieval responses

OUTPUTS:
- agent node behavior assertions

UPSTREAM:
- agent node implementation
- agent state models
- retrieval service
- IndexSnapshot model
- SourceChunk model
- RetrievalResponse model

DOWNSTREAM:
- local test runs
- future CI guardrails
- future LangGraph workflow tests
- future grounded answer tests
- system map review

OWNS:
- planner node tests
- retriever node tests
- verifier node tests
- deterministic plan tests
- planner step status tests
- retriever step status tests
- verifier step status tests
- state copy behavior tests

DOES_NOT_OWN:
- agent state model tests
- responder node tests
- LangGraph integration tests
- LLM answer tests

SIDE_EFFECTS:
- none

STATE:
  reads:
    - in-memory test state
    - in-memory test snapshots
  writes:
    - none

NOTES:
- These tests keep early agent nodes deterministic.
- No LangGraph behavior is expected in this slice.
"""

import pytest

from code_context.agent import (
    AgentStepStatus,
    create_initial_state,
    plan_question,
    retrieve_context_for_question,
    verify_retrieval_for_answer,
)
from code_context.agent.nodes import DEFAULT_CODE_QUESTION_PLAN
from code_context.agent.state import AgentStep, CodeQuestionState
from code_context.index_store import build_index_snapshot
from code_context.models import IndexSnapshot, RetrievalResponse, SourceChunk


def test_plan_question_adds_default_plan_and_completes_planner_step() -> None:
    state = create_initial_state("How does retrieval work?")

    planned = plan_question(state)

    assert planned.plan == DEFAULT_CODE_QUESTION_PLAN
    assert planned.steps[0].name == "planner"
    assert planned.steps[0].status == AgentStepStatus.completed
    assert planned.steps[0].notes == [
        "Created deterministic code question plan.",
        "Question: How does retrieval work?",
    ]


def test_plan_question_returns_copy_without_mutating_original_state() -> None:
    state = create_initial_state("How does scanning work?")

    planned = plan_question(state)

    assert state.plan == []
    assert state.steps[0].status == AgentStepStatus.pending
    assert planned.plan == DEFAULT_CODE_QUESTION_PLAN
    assert planned.steps[0].status == AgentStepStatus.completed


def test_plan_question_replaces_existing_plan_deterministically() -> None:
    state = create_initial_state("How does drift detection work?")
    state_with_old_plan = state.model_copy(update={"plan": ["old plan"]})

    planned = plan_question(state_with_old_plan)

    assert planned.plan == DEFAULT_CODE_QUESTION_PLAN


def test_plan_question_requires_planner_step_to_exist() -> None:
    state = CodeQuestionState(
        question="How does retrieval work?",
        steps=[AgentStep(name="retriever")],
    )

    with pytest.raises(ValueError, match="Unknown agent step"):
        plan_question(state)


def test_retrieve_context_for_question_attaches_sufficient_retrieval() -> None:
    state = create_initial_state("How does calculate file hash work?")
    snapshot = _snapshot(
        chunks=[
            _chunk(
                chunk_id="src/scanner.py:1-2:hash",
                relative_path="src/scanner.py",
                content="def calculate_file_hash(path):\n    return hashlib.sha256(data).hexdigest()\n",
            )
        ]
    )

    updated = retrieve_context_for_question(
        state,
        snapshot=snapshot,
        limit=1,
        minimum_top_score=0.05,
    )

    assert updated.retrieval is not None
    assert updated.retrieval.is_sufficient is True
    assert updated.retrieval.results[0].chunk.relative_path == "src/scanner.py"
    assert updated.steps[1].name == "retriever"
    assert updated.steps[1].status == AgentStepStatus.completed
    assert updated.steps[1].notes == [
        "Retrieved 1 grounded result(s).",
        "Retrieved context passed sufficiency checks.",
    ]


def test_retrieve_context_for_question_records_insufficient_context() -> None:
    state = create_initial_state("How does postgres liquibase migration work?")
    snapshot = _snapshot(
        chunks=[
            _chunk(
                chunk_id="src/api.py:1-2:api",
                relative_path="src/api.py",
                content="FastAPI route definitions only",
            )
        ]
    )

    updated = retrieve_context_for_question(
        state,
        snapshot=snapshot,
        minimum_top_score=0.95,
    )

    assert updated.retrieval is not None
    assert updated.retrieval.is_sufficient is False
    assert (
        updated.retrieval.insufficient_reason
        == "The best retrieved context was below the sufficiency score threshold."
    )
    assert updated.steps[1].status == AgentStepStatus.completed
    assert updated.steps[1].notes == [
        "Retrieved 1 grounded result(s).",
        "Retrieved context was insufficient: "
        "The best retrieved context was below the sufficiency score threshold.",
    ]


def test_retrieve_context_for_question_returns_copy_without_mutating_original_state() -> None:
    state = create_initial_state("How does retrieval work?")
    snapshot = _snapshot(chunks=[_chunk(content="retrieval context chunk")])

    updated = retrieve_context_for_question(state, snapshot=snapshot)

    assert state.retrieval is None
    assert state.steps[1].status == AgentStepStatus.pending
    assert updated.retrieval is not None
    assert updated.steps[1].status == AgentStepStatus.completed


def test_retrieve_context_for_question_requires_retriever_step_to_exist() -> None:
    state = CodeQuestionState(
        question="How does retrieval work?",
        steps=[AgentStep(name="planner")],
    )
    snapshot = _snapshot(chunks=[_chunk(content="retrieval context chunk")])

    with pytest.raises(ValueError, match="Unknown agent step"):
        retrieve_context_for_question(state, snapshot=snapshot)


def test_verify_retrieval_for_answer_marks_sufficient_retrieval_as_answerable() -> None:
    state = create_initial_state("How does calculate file hash work?")
    state_with_retrieval = state.model_copy(
        update={
            "retrieval": RetrievalResponse(
                query="How does calculate file hash work?",
                is_sufficient=True,
                insufficient_reason=None,
                results=[
                    _search_result_chunk(
                        content="def calculate_file_hash(path):\n    return hashlib.sha256(data).hexdigest()\n",
                        relative_path="src/scanner.py",
                    )
                ],
            )
        }
    )

    verified = verify_retrieval_for_answer(state_with_retrieval)

    assert verified.verification is not None
    assert verified.verification.can_answer is True
    assert verified.verification.is_grounded is True
    assert verified.verification.reason is None
    assert verified.steps[2].name == "verifier"
    assert verified.steps[2].status == AgentStepStatus.completed
    assert verified.steps[2].notes == ["Verified retrieval context is sufficient and grounded."]


def test_verify_retrieval_for_answer_marks_missing_retrieval_as_not_answerable() -> None:
    state = create_initial_state("How does retrieval work?")

    verified = verify_retrieval_for_answer(state)

    assert verified.verification is not None
    assert verified.verification.can_answer is False
    assert verified.verification.is_grounded is False
    assert verified.verification.reason == "No retrieval results are attached to the agent state."
    assert verified.steps[2].status == AgentStepStatus.completed
    assert verified.steps[2].notes == [
        "Verification failed: No retrieval results are attached to the agent state."
    ]


def test_verify_retrieval_for_answer_uses_retrieval_insufficient_reason() -> None:
    state = create_initial_state("How does postgres liquibase migration work?")
    state_with_retrieval = state.model_copy(
        update={
            "retrieval": RetrievalResponse(
                query="How does postgres liquibase migration work?",
                is_sufficient=False,
                insufficient_reason="Not enough relevant indexed context was found.",
                results=[],
            )
        }
    )

    verified = verify_retrieval_for_answer(state_with_retrieval)

    assert verified.verification is not None
    assert verified.verification.can_answer is False
    assert verified.verification.is_grounded is False
    assert verified.verification.reason == "Not enough relevant indexed context was found."
    assert verified.steps[2].notes == [
        "Verification failed: Not enough relevant indexed context was found."
    ]


def test_verify_retrieval_for_answer_rejects_sufficient_retrieval_without_results() -> None:
    state = create_initial_state("How does retrieval work?")
    state_with_retrieval = state.model_copy(
        update={
            "retrieval": RetrievalResponse(
                query="How does retrieval work?",
                is_sufficient=True,
                insufficient_reason=None,
                results=[],
            )
        }
    )

    verified = verify_retrieval_for_answer(state_with_retrieval)

    assert verified.verification is not None
    assert verified.verification.can_answer is False
    assert verified.verification.is_grounded is False
    assert verified.verification.reason == "No grounded retrieval results are available."
    assert verified.steps[2].notes == [
        "Verification failed: No grounded retrieval results are available."
    ]


def test_verify_retrieval_for_answer_returns_copy_without_mutating_original_state() -> None:
    state = create_initial_state("How does retrieval work?")

    verified = verify_retrieval_for_answer(state)

    assert state.verification is None
    assert state.steps[2].status == AgentStepStatus.pending
    assert verified.verification is not None
    assert verified.steps[2].status == AgentStepStatus.completed


def test_verify_retrieval_for_answer_requires_verifier_step_to_exist() -> None:
    state = CodeQuestionState(
        question="How does retrieval work?",
        steps=[AgentStep(name="planner"), AgentStep(name="retriever")],
    )

    with pytest.raises(ValueError, match="Unknown agent step"):
        verify_retrieval_for_answer(state)


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
) -> SourceChunk:
    return SourceChunk(
        chunk_id=chunk_id,
        relative_path=relative_path,
        start_line=1,
        end_line=max(1, content.count("\n")),
        content=content,
        language="python",
    )


def _search_result_chunk(*, content: str, relative_path: str) -> dict:
    return {
        "chunk": _chunk(content=content, relative_path=relative_path),
        "score": 0.75,
    }