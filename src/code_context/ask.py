from __future__ import annotations

"""
ROLE: Orchestrate grounded code question answering over an indexed repository snapshot.
LAYER: application_service
FLOW: ask_workflow_orchestration

INPUTS:
- developer question text
- IndexSnapshot records
- retrieval limit
- retrieval score thresholds
- retrieval sufficiency thresholds
- current repository files for stale-index checks
- optional LangGraph runtime availability through the workflow adapter

OUTPUTS:
- AskWorkflowResult records with grounded answers, stale-index refusals, or insufficient-context refusals
- grounded SearchResult records when context is sufficient
- agent step status records when workflow execution reaches the agent

UPSTREAM:
- FastAPI ask endpoint
- future CLI ask command
- future demo scripts
- future LangGraph workflow callers

DOWNSTREAM:
- repository scanner
- drift detector
- optional LangGraph workflow adapter
- deterministic agent workflow fallback
- agent state models
- retrieval result models

OWNS:
- ask workflow orchestration
- ask-time stale-index detection
- stale-index refusal construction
- optional LangGraph adapter invocation
- confidence classification for ask results
- reusable ask result shape for API and future CLI callers

DOES_NOT_OWN:
- HTTP request and response DTOs
- API routing
- repository traversal internals
- file hashing internals
- drift comparison internals
- retrieval scoring
- agent node behavior
- LangGraph graph construction
- LLM prompting

SIDE_EFFECTS:
- reads repository files through scanner for stale-index detection

STATE:
  reads:
    - local repository files
    - in-memory IndexSnapshot records
  writes:
    - none

NOTES:
- Keep ask orchestration independent from FastAPI so CLI and API can share it.
- Refuse before running the agent workflow when indexed context is stale.
- The optional LangGraph adapter owns fallback to the deterministic workflow when LangGraph is unavailable.
"""

from pathlib import Path

from pydantic import BaseModel

from code_context.agent.langgraph_graph import run_code_question_workflow_with_optional_langgraph
from code_context.agent.state import AgentStep
from code_context.drift import detect_drift
from code_context.models import DriftReport, IndexSnapshot, SearchResult
from code_context.retrieval import DEFAULT_MINIMUM_TOP_SCORE, DEFAULT_RETRIEVAL_MIN_SCORE
from code_context.scanner import scan_repository


ASK_STALE_INDEX_REFUSAL = "Indexed context is stale. Re-index the repository before asking questions."


class AskWorkflowResult(BaseModel):
    question: str
    answer: str
    confidence: str
    is_grounded: bool
    is_stale: bool
    insufficient_reason: str | None
    plan: list[str]
    citations: list[str]
    sources: list[SearchResult]
    steps: list[AgentStep]


def ask_indexed_code_question(
    *,
    question: str,
    snapshot: IndexSnapshot,
    limit: int = 5,
    min_score: float = DEFAULT_RETRIEVAL_MIN_SCORE,
    minimum_results: int = 1,
    minimum_top_score: float = DEFAULT_MINIMUM_TOP_SCORE,
    prefer_langgraph: bool = True,
) -> AskWorkflowResult:
    """Answer a code question against an index snapshot or return a refusal."""
    drift_report = detect_snapshot_drift(snapshot)

    if drift_report.is_stale:
        return build_stale_ask_result(question)

    state = run_code_question_workflow_with_optional_langgraph(
        question=question,
        snapshot=snapshot,
        limit=limit,
        min_score=min_score,
        minimum_results=minimum_results,
        minimum_top_score=minimum_top_score,
        prefer_langgraph=prefer_langgraph,
    )

    verification = state.verification
    is_grounded = bool(verification and verification.can_answer and verification.is_grounded)
    confidence = "grounded" if is_grounded else "insufficient_context"
    insufficient_reason = None if is_grounded else verification.reason if verification else None
    sources = state.retrieval.results if state.retrieval else []

    return AskWorkflowResult(
        question=state.question,
        answer=state.answer or "",
        confidence=confidence,
        is_grounded=is_grounded,
        is_stale=False,
        insufficient_reason=insufficient_reason,
        plan=state.plan,
        citations=state.citations,
        sources=sources,
        steps=state.steps,
    )


def detect_snapshot_drift(snapshot: IndexSnapshot) -> DriftReport:
    """Compare a snapshot against the current repository files."""
    current_files = scan_repository(Path(snapshot.repo_root))
    return detect_drift(snapshot=snapshot, current_files=current_files)


def build_stale_ask_result(question: str) -> AskWorkflowResult:
    """Build an ask result that refuses because indexed context is stale."""
    return AskWorkflowResult(
        question=question.strip(),
        answer=(
            "I cannot answer from this index because the indexed context is stale. "
            "Re-index the repository and ask again."
        ),
        confidence="stale_index",
        is_grounded=False,
        is_stale=True,
        insufficient_reason=ASK_STALE_INDEX_REFUSAL,
        plan=[],
        citations=[],
        sources=[],
        steps=[],
    )