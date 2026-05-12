from __future__ import annotations

"""
ROLE: Expose agent workflow state and node helpers for the codebase context assistant.
LAYER: agent
FLOW: agent_package

INPUTS:
- agent state model imports
- agent node helper imports

OUTPUTS:
- public agent package exports

UPSTREAM:
- future LangGraph workflow
- future FastAPI ask endpoint
- future tests

DOWNSTREAM:
- agent state models
- agent nodes
- future agent graph

OWNS:
- public agent package exports

DOES_NOT_OWN:
- retrieval sufficiency logic
- verification behavior
- response generation
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
- Keep this package small until the agent workflow is actually implemented.
"""

from code_context.agent.nodes import plan_question, retrieve_context_for_question
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
    "retrieve_context_for_question",
    "update_step_status",
]