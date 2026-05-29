from __future__ import annotations

"""
ROLE: Suggest deterministic clarification hints for weak fuzzy retrieval.
LAYER: retrieval
FLOW: retrieval_clarification

INPUTS:
- developer question text
- known fuzzy concept trigger phrases
- deterministic related-term suggestion rules

OUTPUTS:
- ClarificationSuggestion records with suggested questions and related terms

UPSTREAM:
- ask workflow orchestration service
- future retrieval alignment checks
- future API ask responses
- future CLI ask output

DOWNSTREAM:
- ask workflow result shaping
- API response models
- CLI terminal formatting
- future clarification loop behavior

OWNS:
- deterministic fuzzy concept detection
- suggested related-term sets
- suggested clarified question text
- clarification reason text

DOES_NOT_OWN:
- source retrieval
- vector scoring
- stale-index detection
- LLM query rewriting
- answer generation
- API routing
- CLI argument parsing

SIDE_EFFECTS:
- none

STATE:
  reads:
    - none
  writes:
    - none

NOTES:
- This module should not call an LLM.
- Suggestions are search aids, not source evidence.
- Keep the first concept library small and explainable.
"""

from collections.abc import Sequence

from pydantic import BaseModel, Field

from code_context.models import SearchResult


class ClarificationSuggestion(BaseModel):
    suggested_question: str
    suggested_related_terms: list[str] = Field(default_factory=list)
    reason: str


_CONCEPT_SUGGESTIONS: tuple[tuple[tuple[str, ...], ClarificationSuggestion], ...] = (
    (
        (
            "another program",
            "talk to this",
            "talk to the",
            "over http",
            "http",
            "outside caller",
            "external caller",
            "web api",
        ),
        ClarificationSuggestion(
            suggested_question="Where is the HTTP API layer implemented, and which routes does it expose?",
            suggested_related_terms=["HTTP", "API", "FastAPI", "endpoint", "route"],
            reason="The question sounds like it may be asking about the HTTP API boundary.",
        ),
    ),
    (
        (
            "auth",
            "authenticate",
            "authentication",
            "authorization",
            "login",
            "session",
            "token",
        ),
        ClarificationSuggestion(
            suggested_question="Where is authentication or authorization handled?",
            suggested_related_terms=[
                "login",
                "password",
                "session",
                "token",
                "security",
                "authentication",
                "authorization",
            ],
            reason="The question sounds like it may be asking about authentication or authorization.",
        ),
    ),
    (
        (
            "saved records",
            "saved metadata",
            "stored records",
            "stored metadata",
            "where are records",
            "where is metadata",
            "local index",
        ),
        ClarificationSuggestion(
            suggested_question="Where are indexed records and metadata persisted?",
            suggested_related_terms=["index", "persistence", "JSON", "snapshot", "metadata"],
            reason="The question sounds like it may be asking about local index persistence.",
        ),
    ),
    (
        (
            "freshness",
            "fresh",
            "stale",
            "out of date",
            "changed files",
            "current files",
            "drift",
        ),
        ClarificationSuggestion(
            suggested_question="Where does the system check whether indexed context is stale?",
            suggested_related_terms=["drift", "stale", "hash", "modified", "timestamp"],
            reason="The question sounds like it may be asking about stale-index or drift detection.",
        ),
    ),
)



def should_suggest_clarification_for_weak_alignment(
    *,
    question: str,
    sources: Sequence[SearchResult],
    related_terms: Sequence[str] | None = None,
) -> bool:
    """Return true when grounded retrieval likely missed the implementation area."""
    if related_terms:
        return False

    if not sources:
        return False

    if suggest_clarification(question) is None:
        return False

    return not any(
        _is_implementation_source_path(source.chunk.relative_path)
        for source in sources
    )


def _is_implementation_source_path(relative_path: str) -> bool:
    normalized_path = relative_path.replace("\\", "/").casefold()

    if normalized_path.startswith("tests/") or "/tests/" in normalized_path:
        return False

    if normalized_path.startswith("docs/"):
        return False

    if ".egg-info/" in normalized_path:
        return False

    if normalized_path.endswith("sources.txt"):
        return False

    return normalized_path.startswith("src/")



def suggest_clarification(question: str) -> ClarificationSuggestion | None:
    """Suggest related terms for known fuzzy question shapes."""
    normalized_question = " ".join(question.casefold().split())
    if not normalized_question:
        return None

    for triggers, suggestion in _CONCEPT_SUGGESTIONS:
        if any(trigger in normalized_question for trigger in triggers):
            return suggestion

    return None
