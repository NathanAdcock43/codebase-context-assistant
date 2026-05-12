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
- grounded retrieval responses
- verification results

OUTPUTS:
- updated CodeQuestionState records
- deterministic plan steps
- grounded retrieval responses attached to state
- verification results attached to state
- grounded or refused answer text attached to state
- grounded citation strings attached to state
- updated planner, retriever, verifier, and responder step statuses

UPSTREAM:
- future LangGraph graph
- future FastAPI ask endpoint
- future agent workflow tests

DOWNSTREAM:
- agent state models
- retrieval service
- future LangGraph workflow
- future API ask response models

OWNS:
- deterministic planner node behavior
- default code question plan creation
- planner step status updates
- retriever node orchestration
- retrieval result attachment to agent state
- retriever step status updates
- verifier node behavior
- verification result attachment to agent state
- verifier step status updates
- deterministic responder node behavior
- grounded citation construction
- refusal response construction

DOES_NOT_OWN:
- retrieval sufficiency checks
- vector scoring implementation
- LLM prompting
- generated natural-language synthesis
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
- The verifier should make a grounded sufficiency decision, not generate the final answer.
- The responder should refuse when verification fails or has not run.
- Future nodes should preserve the same state-copying style.
"""

from code_context.agent.state import (
    AgentStepStatus,
    CodeQuestionState,
    VerificationResult,
    update_step_status,
)
from code_context.models import IndexSnapshot, SearchResult
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


def verify_retrieval_for_answer(state: CodeQuestionState) -> CodeQuestionState:
    """Return state with a verification decision based on attached retrieval context."""
    verification = _build_verification_result(state)
    state_with_verification = state.model_copy(update={"verification": verification})

    if verification.can_answer:
        notes = ["Verified retrieval context is sufficient and grounded."]
    else:
        notes = [f"Verification failed: {verification.reason}"]

    return update_step_status(
        state_with_verification,
        step_name="verifier",
        status=AgentStepStatus.completed,
        notes=notes,
    )


def respond_to_question(state: CodeQuestionState) -> CodeQuestionState:
    """Return state with a deterministic grounded response or refusal."""
    answer, citations, notes = _build_response_parts(state)

    state_with_response = state.model_copy(
        update={
            "answer": answer,
            "citations": citations,
        }
    )

    return update_step_status(
        state_with_response,
        step_name="responder",
        status=AgentStepStatus.completed,
        notes=notes,
    )


def _build_verification_result(state: CodeQuestionState) -> VerificationResult:
    if state.retrieval is None:
        return VerificationResult(
            can_answer=False,
            is_grounded=False,
            reason="No retrieval results are attached to the agent state.",
        )

    if not state.retrieval.is_sufficient:
        return VerificationResult(
            can_answer=False,
            is_grounded=False,
            reason=state.retrieval.insufficient_reason or "Retrieved context was insufficient.",
        )

    if not state.retrieval.results:
        return VerificationResult(
            can_answer=False,
            is_grounded=False,
            reason="No grounded retrieval results are available.",
        )

    return VerificationResult(
        can_answer=True,
        is_grounded=True,
        reason=None,
    )


def _build_response_parts(state: CodeQuestionState) -> tuple[str, list[str], list[str]]:
    if state.verification is None:
        return (
            "I cannot answer because verification has not run.",
            [],
            ["Refused because verification has not run."],
        )

    if not state.verification.can_answer or not state.verification.is_grounded:
        reason = state.verification.reason or "Retrieved context was insufficient."
        return (
            f"I do not have enough grounded indexed context to answer that. Reason: {reason}",
            [],
            [f"Refused because verification failed: {reason}"],
        )

    if state.retrieval is None or not state.retrieval.results:
        reason = "No grounded retrieval results are available."
        return (
            f"I do not have enough grounded indexed context to answer that. Reason: {reason}",
            [],
            [f"Refused because verification failed: {reason}"],
        )

    citations = _build_citations(state.retrieval.results)
    answer = _build_grounded_response(question=state.question, citations=citations)

    return (
        answer,
        citations,
        [f"Created grounded response with {len(citations)} citation(s)."],
    )


def _build_grounded_response(*, question: str, citations: list[str]) -> str:
    citation_text = ", ".join(citations)

    return (
        "I found grounded indexed context for this question. "
        f"Question: {question} "
        f"Relevant source references: {citation_text}"
    )


def _build_citations(results: list[SearchResult]) -> list[str]:
    citations: list[str] = []

    for result in results:
        citation = _format_source_reference(result)
        if citation not in citations:
            citations.append(citation)

    return citations


def _format_source_reference(result: SearchResult) -> str:
    chunk = result.chunk
    return f"{chunk.relative_path}:{chunk.start_line}-{chunk.end_line}"