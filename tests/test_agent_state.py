from __future__ import annotations

"""
ROLE: Verify agent state models and helper behavior.
LAYER: tests
FLOW: agent_state_validation

INPUTS:
- developer question text
- sample retrieval responses
- sample verification results
- agent step status updates

OUTPUTS:
- agent state behavior assertions

UPSTREAM:
- agent state models
- RetrievalResponse model

DOWNSTREAM:
- local test runs
- future CI guardrails
- future LangGraph workflow tests
- future grounded answer tests
- system map review

OWNS:
- initial agent state tests
- step status update tests
- question validation tests
- retrieval attachment tests
- verification attachment tests

DOES_NOT_OWN:
- planner tests
- retriever node tests
- verifier node tests
- responder node tests
- LangGraph integration tests
- LLM answer tests

SIDE_EFFECTS:
- none

STATE:
  reads:
    - in-memory test state
  writes:
    - none

NOTES:
- These tests keep the future agent workflow state explicit and stable.
- No LangGraph behavior is expected in this slice.
"""

import pytest
from pydantic import ValidationError

from code_context.agent import (
    AgentStepStatus,
    CodeQuestionState,
    VerificationResult,
    create_initial_state,
    update_step_status,
)
from code_context.models import RetrievalResponse


def test_create_initial_state_trims_question_and_adds_default_steps() -> None:
    state = create_initial_state("  How does scanning work?  ")

    assert state.question == "How does scanning work?"
    assert state.plan == []
    assert state.retrieval is None
    assert state.verification is None
    assert state.answer is None
    assert state.citations == []
    assert [(step.name, step.status) for step in state.steps] == [
        ("planner", AgentStepStatus.pending),
        ("retriever", AgentStepStatus.pending),
        ("verifier", AgentStepStatus.pending),
        ("responder", AgentStepStatus.pending),
    ]


def test_code_question_state_rejects_empty_question() -> None:
    with pytest.raises(ValidationError, match="question must not be empty"):
        CodeQuestionState(question="   ")


def test_update_step_status_returns_updated_copy() -> None:
    state = create_initial_state("How does retrieval work?")

    updated = update_step_status(
        state,
        step_name="retriever",
        status=AgentStepStatus.completed,
        notes=["Retrieved two grounded chunks."],
    )

    assert state.steps[1].status == AgentStepStatus.pending
    assert updated.steps[1].status == AgentStepStatus.completed
    assert updated.steps[1].notes == ["Retrieved two grounded chunks."]


def test_update_step_status_rejects_unknown_step() -> None:
    state = create_initial_state("How does retrieval work?")

    with pytest.raises(ValueError, match="Unknown agent step"):
        update_step_status(
            state,
            step_name="unknown",
            status=AgentStepStatus.completed,
        )


def test_agent_state_can_store_retrieval_response() -> None:
    retrieval = RetrievalResponse(
        query="hash files",
        is_sufficient=False,
        insufficient_reason="Not enough relevant indexed context was found.",
        results=[],
    )

    state = create_initial_state("How are files hashed?")
    updated = state.model_copy(update={"retrieval": retrieval})

    assert updated.retrieval is not None
    assert updated.retrieval.query == "hash files"
    assert updated.retrieval.is_sufficient is False
    assert updated.retrieval.insufficient_reason == "Not enough relevant indexed context was found."


def test_agent_state_can_store_verification_result_and_refusal_answer() -> None:
    verification = VerificationResult(
        can_answer=False,
        is_grounded=False,
        reason="Retrieved context was insufficient.",
    )

    state = create_initial_state("Explain the payment system.")
    updated = state.model_copy(
        update={
            "verification": verification,
            "answer": "I do not have enough indexed context to answer that.",
            "citations": [],
        }
    )

    assert updated.verification == verification
    assert updated.answer == "I do not have enough indexed context to answer that."
    assert updated.citations == []