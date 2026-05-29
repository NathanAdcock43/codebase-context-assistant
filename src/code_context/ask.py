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
- optional LLM generation request
- optional injected LLM client
- optional LLM model override

OUTPUTS:
- AskWorkflowResult records with grounded answers, stale-index refusals, insufficient-context refusals, optional generated answers, or generated self-refusals
- grounded SearchResult records when context is sufficient
- agent step status records when workflow execution reaches the agent
- optional LLM provider metadata when a generated answer request executes

UPSTREAM:
- FastAPI ask endpoint
- CLI ask command
- future demo scripts
- future LangGraph workflow callers

DOWNSTREAM:
- repository scanner
- drift detector
- optional LangGraph workflow adapter
- deterministic agent workflow fallback
- agent state models
- retrieval result models
- configured LLM client factory
- grounded answer generation service

OWNS:
- ask workflow orchestration
- ask-time stale-index detection
- stale-index refusal construction
- optional LangGraph adapter invocation
- confidence classification for ask results
- reusable ask result shape for API and CLI callers
- optional generated answer orchestration after grounding checks pass
- guardrails that prevent LLM calls when context is stale or insufficient
- generated-answer self-refusal confidence downgrades

DOES_NOT_OWN:
- HTTP request and response DTOs
- API routing
- CLI argument parsing
- repository traversal internals
- file hashing internals
- drift comparison internals
- retrieval scoring
- agent node behavior
- LangGraph graph construction
- prompt formatting internals
- provider-specific SDK calls

SIDE_EFFECTS:
- reads repository files through scanner for stale-index detection
- may call an injected or configured LLM client when use_llm is true and context is grounded

STATE:
  reads:
    - local repository files
    - in-memory IndexSnapshot records
    - process environment when a configured LLM client or default model is needed
  writes:
    - none

NOTES:
- Keep ask orchestration independent from FastAPI so CLI and API can share it.
- Refuse before running the agent workflow when indexed context is stale.
- Do not call an LLM when retrieval or verification says context is insufficient.
- Generated answers that report insufficient supplied context should not keep grounded_generated confidence.
- The optional LangGraph adapter owns fallback to the deterministic workflow when LangGraph is unavailable.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from code_context.agent.langgraph_graph import run_code_question_workflow_with_optional_langgraph
from code_context.agent.state import AgentStep
from code_context.answer_generation import generate_grounded_answer
from code_context.clarification import (
    should_suggest_clarification_for_weak_alignment,
    suggest_clarification,
)
from code_context.config import load_app_config
from code_context.drift import detect_drift
from code_context.llm import LlmClient
from code_context.llm_factory import build_configured_llm_client
from code_context.models import DriftReport, IndexSnapshot, SearchResult
from code_context.retrieval import (
    DEFAULT_MINIMUM_TOP_SCORE,
    DEFAULT_RETRIEVAL_MIN_SCORE,
    build_query_with_related_terms,
    find_matched_related_terms,
    normalize_related_terms,
)
from code_context.scanner import scan_repository


ASK_STALE_INDEX_REFUSAL = "Indexed context is stale. Re-index the repository before asking questions."
LLM_GENERATED_CONFIDENCE = "grounded_generated"
LLM_GENERATED_REFUSAL_CONFIDENCE = "generated_refusal"
GENERATED_ANSWER_REFUSAL_REASON = (
    "Generated answer reported that the supplied sources were insufficient."
)
GENERATED_ANSWER_REFUSAL_MARKERS = (
    "provided sources are insufficient",
    "provided source context is insufficient",
    "supplied sources are insufficient",
    "supplied source context is insufficient",
    "current indexed context is insufficient",
    "provided context is insufficient",
    "source context is insufficient",
    "not present in the supplied context",
    "not present in the provided context",
    "not present in the indexed context",
    "not included in the provided sources",
    "not included in the supplied sources",
    "not included in the indexed context",
    "not shown in the provided source context",
    "not shown in the provided sources",
    "not shown in the indexed context",
    "implementation is not shown",
    "implementation details are not included",
    "provided context does not include",
    "cannot answer from the provided sources",
    "can't answer from the provided sources",
    "insufficient context",
)


class AskWorkflowResult(BaseModel):
    question: str
    related_terms: list[str] = Field(default_factory=list)
    matched_related_terms: list[str] = Field(default_factory=list)
    answer: str
    confidence: str
    is_grounded: bool
    is_stale: bool
    insufficient_reason: str | None
    plan: list[str]
    citations: list[str]
    sources: list[SearchResult]
    steps: list[AgentStep]
    is_llm_generated: bool = False
    needs_clarification: bool = False
    suggested_question: str | None = None
    suggested_related_terms: list[str] = Field(default_factory=list)
    clarification_reason: str | None = None
    llm_provider: str | None = None
    llm_model: str | None = None
    llm_usage: dict[str, int] = Field(default_factory=dict)


def ask_indexed_code_question(
    *,
    question: str,
    snapshot: IndexSnapshot,
    related_terms: Sequence[str] | None = None,
    limit: int = 5,
    min_score: float = DEFAULT_RETRIEVAL_MIN_SCORE,
    minimum_results: int = 1,
    minimum_top_score: float = DEFAULT_MINIMUM_TOP_SCORE,
    prefer_langgraph: bool = True,
    use_llm: bool = False,
    llm_client: LlmClient | None = None,
    llm_model: str | None = None,
    llm_temperature: float = 0.0,
) -> AskWorkflowResult:
    """Answer a code question against an index snapshot or return a refusal."""
    normalized_question = question.strip()
    normalized_related_terms = normalize_related_terms(related_terms)
    workflow_question = build_query_with_related_terms(
        normalized_question,
        normalized_related_terms,
    )

    drift_report = detect_snapshot_drift(snapshot)

    if drift_report.is_stale:
        return build_stale_ask_result(
            normalized_question,
            related_terms=normalized_related_terms,
        )

    state = run_code_question_workflow_with_optional_langgraph(
        question=workflow_question,
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

    result = AskWorkflowResult(
        question=normalized_question,
        related_terms=normalized_related_terms,
        matched_related_terms=find_matched_related_terms(sources, normalized_related_terms),
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

    if is_grounded and should_suggest_clarification_for_weak_alignment(
        question=normalized_question,
        sources=sources,
        related_terms=normalized_related_terms,
    ):
        clarification_result = build_clarification_ask_result(
            result,
            answer_intro=(
                "I found related indexed context, but it may not be the implementation area you meant."
            ),
            reason_suffix=(
                "Retrieved context did not include implementation source files."
            ),
        )
        if clarification_result is not None:
            return clarification_result

    if not is_grounded and not normalized_related_terms:
        clarification_result = build_clarification_ask_result(result)
        if clarification_result is not None:
            return clarification_result

    if not use_llm or not is_grounded:
        return result

    return build_generated_ask_result(
        result=result,
        llm_client=llm_client,
        llm_model=llm_model,
        llm_temperature=llm_temperature,
    )


def build_clarification_ask_result(
    result: AskWorkflowResult,
    *,
    answer_intro: str | None = None,
    reason_suffix: str | None = None,
) -> AskWorkflowResult | None:
    """Add a deterministic clarification suggestion when retrieval likely missed intent."""
    suggestion = suggest_clarification(result.question)
    if suggestion is None:
        return None

    related_terms_text = ", ".join(suggestion.suggested_related_terms)
    intro = (
        answer_intro
        or "I found insufficient indexed context for that wording, but the question may be fuzzy."
    )
    clarification_reason = suggestion.reason
    if reason_suffix:
        clarification_reason = f"{clarification_reason} {reason_suffix}"

    return result.model_copy(
        update={
            "answer": (
                f"{intro} "
                f"Try asking: \"{suggestion.suggested_question}\" "
                f"with related terms: {related_terms_text}."
            ),
            "confidence": "needs_clarification",
            "is_grounded": False,
            "needs_clarification": True,
            "suggested_question": suggestion.suggested_question,
            "suggested_related_terms": suggestion.suggested_related_terms,
            "clarification_reason": clarification_reason,
        }
    )



def build_generated_ask_result(
    *,
    result: AskWorkflowResult,
    llm_client: LlmClient | None = None,
    llm_model: str | None = None,
    llm_temperature: float = 0.0,
) -> AskWorkflowResult:
    """Generate a grounded answer from already-verified source context."""
    if not result.sources:
        return result

    resolved_client = llm_client or build_configured_llm_client()
    resolved_model = llm_model or load_app_config().openai_model

    generated_answer = generate_grounded_answer(
        question=result.question,
        results=result.sources,
        client=resolved_client,
        model=resolved_model,
        temperature=llm_temperature,
    )

    generated_answer_text = _generated_answer_text(generated_answer)
    is_generated_refusal = _generated_answer_reports_insufficient_context(generated_answer_text)

    return result.model_copy(
        update={
            "answer": generated_answer_text,
            "confidence": (
                LLM_GENERATED_REFUSAL_CONFIDENCE
                if is_generated_refusal
                else LLM_GENERATED_CONFIDENCE
            ),
            "is_grounded": False if is_generated_refusal else result.is_grounded,
            "insufficient_reason": (
                GENERATED_ANSWER_REFUSAL_REASON
                if is_generated_refusal
                else result.insufficient_reason
            ),
            "is_llm_generated": True,
            "llm_provider": _generated_answer_provider(generated_answer),
            "llm_model": _generated_answer_model(generated_answer, fallback_model=resolved_model),
            "llm_usage": _generated_answer_usage(generated_answer),
        }
    )


def detect_snapshot_drift(snapshot: IndexSnapshot) -> DriftReport:
    """Compare a snapshot against the current repository files."""
    current_files = scan_repository(Path(snapshot.repo_root))
    return detect_drift(snapshot=snapshot, current_files=current_files)


def build_stale_ask_result(
    question: str,
    *,
    related_terms: Sequence[str] | None = None,
) -> AskWorkflowResult:
    """Build an ask result that refuses because indexed context is stale."""
    return AskWorkflowResult(
        question=question.strip(),
        related_terms=normalize_related_terms(related_terms),
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


def _generated_answer_reports_insufficient_context(answer_text: str) -> bool:
    """Return true when generated text says the supplied context is insufficient."""
    normalized_answer = " ".join(answer_text.casefold().split())
    return any(marker in normalized_answer for marker in GENERATED_ANSWER_REFUSAL_MARKERS)


def _generated_answer_text(generated_answer: Any) -> str:
    answer = getattr(generated_answer, "answer", None)
    if isinstance(answer, str):
        return answer

    content = getattr(generated_answer, "content", None)
    if isinstance(content, str):
        return content

    return str(generated_answer)


def _generated_answer_provider(generated_answer: Any) -> str | None:
    provider = getattr(generated_answer, "provider", None)
    return provider if isinstance(provider, str) else None


def _generated_answer_model(generated_answer: Any, *, fallback_model: str) -> str:
    model = getattr(generated_answer, "model", None)
    return model if isinstance(model, str) and model else fallback_model


def _generated_answer_usage(generated_answer: Any) -> dict[str, int]:
    usage = getattr(generated_answer, "usage", None)
    if not isinstance(usage, dict):
        return {}

    normalized_usage: dict[str, int] = {}
    for key, value in usage.items():
        if isinstance(key, str) and isinstance(value, int):
            normalized_usage[key] = value

    return normalized_usage
