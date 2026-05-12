from __future__ import annotations

"""
ROLE: Provide an optional LangGraph adapter for the code question workflow.
LAYER: agent
FLOW: langgraph_agent_workflow_adapter

INPUTS:
- developer question text
- IndexSnapshot records
- retrieval limit
- retrieval score thresholds
- retrieval sufficiency thresholds
- optional LangGraph runtime dependency

OUTPUTS:
- completed CodeQuestionState records with plan, retrieval, verification, answer, citations, and step statuses
- compiled LangGraph workflows when LangGraph is installed
- deterministic fallback workflow results when requested or when LangGraph is unavailable

UPSTREAM:
- future FastAPI ask endpoint
- future CLI ask command
- future LangGraph workflow tests
- future demo scripts

DOWNSTREAM:
- agent node functions
- deterministic workflow runner
- agent state models
- retrieval service
- future API ask response models

OWNS:
- optional LangGraph availability detection
- LangGraph workflow construction
- LangGraph node wrapping around existing deterministic node functions
- deterministic fallback selection for local development
- LangGraph-specific workflow execution

DOES_NOT_OWN:
- planner node internals
- retrieval execution internals
- retrieval sufficiency checks
- verification decision internals
- response construction internals
- API routing
- LLM prompting

SIDE_EFFECTS:
- none

STATE:
  reads:
    - in-memory index snapshots
  writes:
    - none

NOTES:
- Keep this as an adapter around existing tested nodes.
- Do not replace the deterministic graph runner until the LangGraph path is proven.
- This module should allow the project to keep passing locally even before LangGraph is installed.
"""

from typing import Any, TypedDict

from code_context.agent.graph import run_code_question_workflow
from code_context.agent.nodes import (
    plan_question,
    respond_to_question,
    retrieve_context_for_question,
    verify_retrieval_for_answer,
)
from code_context.agent.state import CodeQuestionState, create_initial_state
from code_context.models import IndexSnapshot
from code_context.retrieval import DEFAULT_MINIMUM_TOP_SCORE, DEFAULT_RETRIEVAL_MIN_SCORE

try:
    from langgraph.graph import END, StateGraph
except ImportError as exc:
    END = None
    StateGraph = None
    _LANGGRAPH_IMPORT_ERROR: ImportError | None = exc
else:
    _LANGGRAPH_IMPORT_ERROR = None


class LangGraphCodeQuestionState(TypedDict):
    question: str
    snapshot: IndexSnapshot
    limit: int
    min_score: float
    minimum_results: int
    minimum_top_score: float
    agent_state: CodeQuestionState | None


def is_langgraph_available() -> bool:
    """Return whether the optional LangGraph runtime is importable."""
    return StateGraph is not None and END is not None


def build_langgraph_code_question_workflow() -> Any:
    """Build and compile the LangGraph workflow around the existing agent nodes."""
    if not is_langgraph_available():
        raise RuntimeError(
            "LangGraph is not installed. Install the langgraph dependency to use the LangGraph workflow."
        ) from _LANGGRAPH_IMPORT_ERROR

    workflow = StateGraph(LangGraphCodeQuestionState)
    workflow.add_node("planner", _planner_node)
    workflow.add_node("retriever", _retriever_node)
    workflow.add_node("verifier", _verifier_node)
    workflow.add_node("responder", _responder_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "retriever")
    workflow.add_edge("retriever", "verifier")
    workflow.add_edge("verifier", "responder")
    workflow.add_edge("responder", END)

    return workflow.compile()


def run_langgraph_code_question_workflow(
    *,
    question: str,
    snapshot: IndexSnapshot,
    limit: int = 5,
    min_score: float = DEFAULT_RETRIEVAL_MIN_SCORE,
    minimum_results: int = 1,
    minimum_top_score: float = DEFAULT_MINIMUM_TOP_SCORE,
) -> CodeQuestionState:
    """Run the LangGraph code question workflow and return the final agent state."""
    workflow = build_langgraph_code_question_workflow()

    result = workflow.invoke(
        LangGraphCodeQuestionState(
            question=question,
            snapshot=snapshot,
            limit=limit,
            min_score=min_score,
            minimum_results=minimum_results,
            minimum_top_score=minimum_top_score,
            agent_state=None,
        )
    )

    agent_state = result.get("agent_state")

    if agent_state is None:
        raise RuntimeError("LangGraph workflow completed without an agent state.")

    return agent_state


def run_code_question_workflow_with_optional_langgraph(
    *,
    question: str,
    snapshot: IndexSnapshot,
    limit: int = 5,
    min_score: float = DEFAULT_RETRIEVAL_MIN_SCORE,
    minimum_results: int = 1,
    minimum_top_score: float = DEFAULT_MINIMUM_TOP_SCORE,
    prefer_langgraph: bool = True,
) -> CodeQuestionState:
    """Run LangGraph when available, otherwise use the deterministic workflow runner."""
    if prefer_langgraph and is_langgraph_available():
        return run_langgraph_code_question_workflow(
            question=question,
            snapshot=snapshot,
            limit=limit,
            min_score=min_score,
            minimum_results=minimum_results,
            minimum_top_score=minimum_top_score,
        )

    return run_code_question_workflow(
        question=question,
        snapshot=snapshot,
        limit=limit,
        min_score=min_score,
        minimum_results=minimum_results,
        minimum_top_score=minimum_top_score,
    )


def _planner_node(payload: LangGraphCodeQuestionState) -> dict[str, CodeQuestionState]:
    state = create_initial_state(payload["question"])
    return {"agent_state": plan_question(state)}


def _retriever_node(payload: LangGraphCodeQuestionState) -> dict[str, CodeQuestionState]:
    state = _require_agent_state(payload)

    return {
        "agent_state": retrieve_context_for_question(
            state,
            snapshot=payload["snapshot"],
            limit=payload["limit"],
            min_score=payload["min_score"],
            minimum_results=payload["minimum_results"],
            minimum_top_score=payload["minimum_top_score"],
        )
    }


def _verifier_node(payload: LangGraphCodeQuestionState) -> dict[str, CodeQuestionState]:
    return {"agent_state": verify_retrieval_for_answer(_require_agent_state(payload))}


def _responder_node(payload: LangGraphCodeQuestionState) -> dict[str, CodeQuestionState]:
    return {"agent_state": respond_to_question(_require_agent_state(payload))}


def _require_agent_state(payload: LangGraphCodeQuestionState) -> CodeQuestionState:
    agent_state = payload.get("agent_state")

    if agent_state is None:
        raise RuntimeError("LangGraph workflow node requires an agent state.")

    return agent_state