from __future__ import annotations

"""
ROLE: Expose agent workflow state, node helpers, and deterministic workflow runner.
LAYER: agent
FLOW: agent_package

INPUTS:
- agent state model imports
- agent node helper imports
- agent workflow runner imports

OUTPUTS:
- public agent package exports

UPSTREAM:
- future LangGraph workflow
- future FastAPI ask endpoint
- future tests

DOWNSTREAM:
- agent state models
- agent nodes
- deterministic workflow runner
- future agent graph

OWNS:
- public agent package exports

DOES_NOT_OWN:
- retrieval sufficiency logic
- verification decision internals
- response generation internals
- LangGraph wiring
- API routing

SIDE_EFFECTS:
- none

STATE:
  reads:
    - none
  writes:
    - none

NOTES:
- Keep this package small until the LangGraph workflow is actually implemented.
"""

from code_context.agent.graph import run_code_question_workflow
from code_context.agent.nodes import (
    plan_question,
    respond_to_question,
    retrieve_context_for_question,
    verify_retrieval_for_answer,
)
from code_context.agent.state import (
    AgentStep,
    AgentStepStatus,
    CodeQuestionState,
    VerificationResult,
    create_initial_state,
    update_step_status,
)

__all__ = [
    "AgentStep",
    "AgentStepStatus",
    "CodeQuestionState",
    "VerificationResult",
    "create_initial_state",
    "plan_question",
    "respond_to_question",
    "retrieve_context_for_question",
    "run_code_question_workflow",
    "update_step_status",
    "verify_retrieval_for_answer",
]