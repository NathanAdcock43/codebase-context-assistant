from __future__ import annotations

"""
ROLE: Define state models for the future code question agent workflow.
LAYER: agent
FLOW: agent_state

INPUTS:
- developer question text
- planned workflow steps
- grounded retrieval response
- verification result
- generated or refused answer text
- grounded citation strings

OUTPUTS:
- CodeQuestionState
- AgentStep
- VerificationResult

UPSTREAM:
- future FastAPI ask endpoint
- future LangGraph graph
- future planner node
- future retriever node
- future verifier node
- future responder node

DOWNSTREAM:
- future agent graph
- future agent nodes
- future answer generation
- future grounding tests
- future API response models

OWNS:
- agent state shape
- agent step status shape
- verification result shape
- initial state construction
- step status update helper

DOES_NOT_OWN:
- planning logic
- retrieval logic
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
- This slice defines structure only.
- Keep the state explicit so the workflow remains explainable.
- The future agent should refuse when retrieval or verification says context is insufficient.
"""

from enum import Enum

from pydantic import BaseModel, Field, field_validator

from code_context.models import RetrievalResponse


class AgentStepStatus(str, Enum):
    pending = "pending"
    completed = "completed"
    skipped = "skipped"
    failed = "failed"


class AgentStep(BaseModel):
    name: str
    status: AgentStepStatus = Field(default=AgentStepStatus.pending)
    notes: list[str] = Field(default_factory=list)


class VerificationResult(BaseModel):
    can_answer: bool
    is_grounded: bool
    reason: str | None = Field(default=None)


class CodeQuestionState(BaseModel):
    question: str
    plan: list[str] = Field(default_factory=list)
    retrieval: RetrievalResponse | None = Field(default=None)
    verification: VerificationResult | None = Field(default=None)
    answer: str | None = Field(default=None)
    citations: list[str] = Field(default_factory=list)
    steps: list[AgentStep] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, value: str) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError("question must not be empty")

        return normalized


def create_initial_state(question: str) -> CodeQuestionState:
    """Create a new agent state with the default explainable workflow steps."""
    return CodeQuestionState(
        question=question,
        steps=[
            AgentStep(name="planner"),
            AgentStep(name="retriever"),
            AgentStep(name="verifier"),
            AgentStep(name="responder"),
        ],
    )


def update_step_status(
    state: CodeQuestionState,
    *,
    step_name: str,
    status: AgentStepStatus,
    notes: list[str] | None = None,
) -> CodeQuestionState:
    """Return a copied state with one workflow step status updated."""
    updated_steps: list[AgentStep] = []
    found_step = False

    for step in state.steps:
        if step.name != step_name:
            updated_steps.append(step)
            continue

        found_step = True
        updated_steps.append(
            step.model_copy(
                update={
                    "status": status,
                    "notes": notes if notes is not None else step.notes,
                }
            )
        )

    if not found_step:
        raise ValueError(f"Unknown agent step: {step_name}")

    return state.model_copy(update={"steps": updated_steps})