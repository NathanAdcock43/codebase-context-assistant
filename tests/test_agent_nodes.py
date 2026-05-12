from __future__ import annotations

"""
ROLE: Verify deterministic agent node behavior.
LAYER: tests
FLOW: agent_node_validation

INPUTS:
- CodeQuestionState records
- developer question text
- existing agent steps

OUTPUTS:
- agent node behavior assertions

UPSTREAM:
- agent node implementation
- agent state models

DOWNSTREAM:
- local test runs
- future CI guardrails
- future LangGraph workflow tests
- future grounded answer tests
- system map review

OWNS:
- planner node tests
- deterministic plan tests
- planner step status tests
- state copy behavior tests

DOES_NOT_OWN:
- agent state model tests
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
- These tests keep the first agent node deterministic.
- No LangGraph behavior is expected in this slice.
"""

import pytest

from code_context.agent import AgentStepStatus, create_initial_state, plan_question
from code_context.agent.nodes import DEFAULT_CODE_QUESTION_PLAN
from code_context.agent.state import AgentStep, CodeQuestionState


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