from __future__ import annotations

"""
ROLE: Define deterministic agent node functions for the code question workflow.
LAYER: agent
FLOW: agent_nodes

INPUTS:
- CodeQuestionState records
- developer question text
- existing agent step status records

OUTPUTS:
- updated CodeQuestionState records
- deterministic plan steps
- updated planner step status

UPSTREAM:
- future LangGraph graph
- future FastAPI ask endpoint
- future agent workflow tests

DOWNSTREAM:
- agent state models
- future retriever node
- future verifier node
- future responder node
- future LangGraph workflow

OWNS:
- deterministic planner node behavior
- default code question plan creation
- planner step status updates

DOES_NOT_OWN:
- retrieval execution
- retrieval sufficiency checks
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
  writes:
    - none

NOTES:
- Keep node behavior deterministic until the graph wiring is ready.
- The planner should produce an explainable plan, not a generated answer.
- Future nodes should preserve the same state-copying style.
"""

from code_context.agent.state import AgentStepStatus, CodeQuestionState, update_step_status


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