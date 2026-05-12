from __future__ import annotations

"""
ROLE: Define deterministic agent node functions for the code question workflow.
LAYER: agent
FLOW: agent_nodes

INPUTS:
- CodeQuestionState records
- developer question text
- existing agent step status records
- IndexSnapshot records
- retrieval thresholds

OUTPUTS:
- updated CodeQuestionState records
- deterministic plan steps
- grounded retrieval responses attached to state
- updated planner and retriever step statuses

UPSTREAM:
- future LangGraph graph
- future FastAPI ask endpoint
- future agent workflow tests

DOWNSTREAM:
- agent state models
- retrieval service
- future verifier node
- future responder node
- future LangGraph workflow

OWNS:
- deterministic planner node behavior
- default code question plan creation
- planner step status updates
- retriever node orchestration
- retrieval result attachment to agent state
- retriever step status updates

DOES_NOT_OWN:
- retrieval sufficiency checks
- vector scoring implementation
- verification logic
- response generation
- LLM prompting
- LangGraph edge definitions
- API routing

SIDE_EFFECTS:
- none

STATE:
  reads:
    - in-memory agent state
    - in-memory index snapshots
  writes:
    - none

NOTES:
- Keep node behavior deterministic until the graph wiring is ready.
- The planner should produce an explainable plan, not a generated answer.
- The retriever should attach grounded context, not decide the final answer.
- Future nodes should preserve the same state-copying style.
"""

from code_context.agent.state import AgentStepStatus, CodeQuestionState, update_step_status
from code_context.models import IndexSnapshot
from code_context.retrieval import (
    DEFAULT_MINIMUM_TOP_SCORE,
    DEFAULT_RETRIEVAL_MIN_SCORE,
    retrieve_grounded_context,
)


DEFAULT_CODE_QUESTION_PLAN = [
    "Clarify the developer question into a retrieval target.",
    "Retrieve grounded source chunks from the local index.",
    "Verify that retrieved context is sufficient and grounded.",
    "Respond with a grounded answer or refuse if context is insufficient.",
]


def plan_question(state: CodeQuestionState) -> CodeQuestionState:
    """Return state with a deterministic plan and completed planner step."""
    planned_state = state.model_copy(update={"plan": list(DEFAULT_CODE_QUESTION_PLAN)})

    return update_step_status(
        planned_state,
        step_name="planner",
        status=AgentStepStatus.completed,
        notes=[
            "Created deterministic code question plan.",
            f"Question: {planned_state.question}",
        ],
    )


def retrieve_context_for_question(
    state: CodeQuestionState,
    *,
    snapshot: IndexSnapshot,
    limit: int = 5,
    min_score: float = DEFAULT_RETRIEVAL_MIN_SCORE,
    minimum_results: int = 1,
    minimum_top_score: float = DEFAULT_MINIMUM_TOP_SCORE,
) -> CodeQuestionState:
    """Return state with grounded retrieval results attached."""
    retrieval_response = retrieve_grounded_context(
        snapshot=snapshot,
        query=state.question,
        limit=limit,
        min_score=min_score,
        minimum_results=minimum_results,
        minimum_top_score=minimum_top_score,
    )

    notes = [f"Retrieved {len(retrieval_response.results)} grounded result(s)."]

    if retrieval_response.is_sufficient:
        notes.append("Retrieved context passed sufficiency checks.")
    else:
        notes.append(
            f"Retrieved context was insufficient: {retrieval_response.insufficient_reason}"
        )

    state_with_retrieval = state.model_copy(update={"retrieval": retrieval_response})

    return update_step_status(
        state_with_retrieval,
        step_name="retriever",
        status=AgentStepStatus.completed,
        notes=notes,
    )