from __future__ import annotations

"""
ROLE: Retrieve grounded source context from an index snapshot and decide whether it is sufficient.
LAYER: retrieval
FLOW: grounded_context_retrieval

INPUTS:
- IndexSnapshot records
- developer query text
- search limit
- search score threshold
- minimum result count
- minimum top score threshold

OUTPUTS:
- RetrievalResponse records with sufficiency decisions and grounded SearchResult records

UPSTREAM:
- JSON index store
- FastAPI retrieve endpoint
- future retriever agent node
- future answer generation workflow

DOWNSTREAM:
- verifier
- responder
- FastAPI response models
- future LangGraph workflow
- grounded answer generation

OWNS:
- retrieval orchestration over indexed chunks
- retrieval sufficiency checks
- insufficient-context refusal reasons
- loading indexed snapshots for retrieval
- preserving grounded SearchResult records

DOES_NOT_OWN:
- repository scanning
- source chunking
- JSON persistence internals
- vector scoring implementation
- drift detection
- LLM prompting
- answer generation
- API route definitions

SIDE_EFFECTS:
- load_and_retrieve_context reads a local JSON index file

STATE:
  reads:
    - local JSON index file when load_and_retrieve_context is used
  writes:
    - none

NOTES:
- This service does not generate answers.
- It decides whether enough grounded context exists for a future answer step.
- Keep refusal reasons plain and developer-facing.
"""

from pathlib import Path

from code_context.index_store import DEFAULT_INDEX_FILENAME, JsonIndexStore
from code_context.models import IndexSnapshot, RetrievalResponse
from code_context.vector_store import search_chunks


DEFAULT_RETRIEVAL_MIN_SCORE = 0.001
DEFAULT_MINIMUM_TOP_SCORE = 0.05


def load_and_retrieve_context(
    *,
    index_dir: Path | str,
    query: str,
    limit: int = 5,
    min_score: float = DEFAULT_RETRIEVAL_MIN_SCORE,
    minimum_results: int = 1,
    minimum_top_score: float = DEFAULT_MINIMUM_TOP_SCORE,
    index_filename: str = DEFAULT_INDEX_FILENAME,
) -> RetrievalResponse:
    """Load an index snapshot and retrieve grounded context from it."""
    snapshot = JsonIndexStore(index_dir, index_filename=index_filename).load()

    return retrieve_grounded_context(
        snapshot=snapshot,
        query=query,
        limit=limit,
        min_score=min_score,
        minimum_results=minimum_results,
        minimum_top_score=minimum_top_score,
    )


def retrieve_grounded_context(
    *,
    snapshot: IndexSnapshot,
    query: str,
    limit: int = 5,
    min_score: float = DEFAULT_RETRIEVAL_MIN_SCORE,
    minimum_results: int = 1,
    minimum_top_score: float = DEFAULT_MINIMUM_TOP_SCORE,
) -> RetrievalResponse:
    """Retrieve relevant chunks and report whether the context is sufficient."""
    _validate_retrieval_settings(
        limit=limit,
        min_score=min_score,
        minimum_results=minimum_results,
        minimum_top_score=minimum_top_score,
    )

    normalized_query = query.strip()

    if not normalized_query:
        return RetrievalResponse(
            query=normalized_query,
            is_sufficient=False,
            insufficient_reason="Query is empty.",
            results=[],
        )

    if not snapshot.chunks:
        return RetrievalResponse(
            query=normalized_query,
            is_sufficient=False,
            insufficient_reason="The index does not contain any chunks to search.",
            results=[],
        )

    results = search_chunks(
        snapshot.chunks,
        normalized_query,
        limit=limit,
        min_score=min_score,
    )

    insufficient_reason = _determine_insufficient_reason(
        results=results,
        minimum_results=minimum_results,
        minimum_top_score=minimum_top_score,
    )

    return RetrievalResponse(
        query=normalized_query,
        is_sufficient=insufficient_reason is None,
        insufficient_reason=insufficient_reason,
        results=results,
    )


def _determine_insufficient_reason(
    *,
    results: list,
    minimum_results: int,
    minimum_top_score: float,
) -> str | None:
    if len(results) < minimum_results:
        return "Not enough relevant indexed context was found."

    top_score = results[0].score if results else 0.0
    if top_score < minimum_top_score:
        return "The best retrieved context was below the sufficiency score threshold."

    return None


def _validate_retrieval_settings(
    *,
    limit: int,
    min_score: float,
    minimum_results: int,
    minimum_top_score: float,
) -> None:
    if limit < 1:
        raise ValueError("limit must be at least 1")

    if min_score < 0:
        raise ValueError("min_score must be 0 or greater")

    if minimum_results < 1:
        raise ValueError("minimum_results must be at least 1")

    if minimum_results > limit:
        raise ValueError("minimum_results must be less than or equal to limit")

    if minimum_top_score < 0:
        raise ValueError("minimum_top_score must be 0 or greater")