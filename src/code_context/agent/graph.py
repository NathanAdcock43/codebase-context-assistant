from __future__ import annotations

"""
ROLE: Run the deterministic code question workflow across planner, retriever, verifier, and responder nodes.
LAYER: agent
FLOW: deterministic_agent_workflow

INPUTS:
- developer question text
- IndexSnapshot records
- retrieval limit
- retrieval score thresholds
- retrieval sufficiency thresholds

OUTPUTS:
- completed CodeQuestionState records with plan, retrieval, verification, answer, citations, and step statuses

UPSTREAM:
- future FastAPI ask endpoint
- future CLI ask command
- future LangGraph replacement workflow
- agent workflow tests

DOWNSTREAM:
- agent node functions
- agent state models
- retrieval service
- future API ask response models

OWNS:
- deterministic workflow ordering
- planner to retriever to verifier to responder orchestration
- workflow-level input validation through CodeQuestionState construction
- final CodeQuestionState return value

DOES_NOT_OWN:
- planner node internals
- retrieval execution internals
- retrieval sufficiency checks
- verification decision internals
- response construction internals
- LLM prompting
- LangGraph edge definitions
- API routing

SIDE_EFFECTS:
- none

STATE:
  reads:
    - in-memory index snapshots
  writes:
    - none

NOTES:
- This runner gives us an explainable local workflow before adding LangGraph.
- LangGraph can later wrap or replace this orchestration while preserving the same node behavior.
- Keep the workflow order explicit.
"""

from code_context.agent.nodes import (
    plan_question,
    respond_to_question,
    retrieve_context_for_question,
    verify_retrieval_for_answer,
)
from code_context.agent.state import CodeQuestionState, create_initial_state
from code_context.models import IndexSnapshot
from code_context.retrieval import DEFAULT_MINIMUM_TOP_SCORE, DEFAULT_RETRIEVAL_MIN_SCORE


def run_code_question_workflow(
    *,
    question: str,
    snapshot: IndexSnapshot,
    limit: int = 5,
    min_score: float = DEFAULT_RETRIEVAL_MIN_SCORE,
    minimum_results: int = 1,
    minimum_top_score: float = DEFAULT_MINIMUM_TOP_SCORE,
) -> CodeQuestionState:
    """Run the deterministic code question workflow and return final state."""
    state = create_initial_state(question)
    state = plan_question(state)
    state = retrieve_context_for_question(
        state,
        snapshot=snapshot,
        limit=limit,
        min_score=min_score,
        minimum_results=minimum_results,
        minimum_top_score=minimum_top_score,
    )
    state = verify_retrieval_for_answer(state)
    return respond_to_question(state)