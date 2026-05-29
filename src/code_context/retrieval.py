from __future__ import annotations

"""
ROLE: Retrieve grounded source context from an index snapshot and decide whether it is sufficient.
LAYER: retrieval
FLOW: grounded_context_retrieval

INPUTS:
- IndexSnapshot records
- developer query text
- deterministic enriched retrieval query text
- path-like query fragments when a question names a specific source file
- deterministic query anchors for fuzzy ticket-style questions
- search limit
- search score threshold
- minimum result count
- minimum top score threshold

OUTPUTS:
- RetrievalResponse records with sufficiency decisions and grounded SearchResult records
- pruned retrieval results that drop weak trailing matches
- insufficient-context refusals when a named source path is missing from retrieved results

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
- deterministic retrieval query enrichment before vector search
- retrieval sufficiency checks
- weak trailing result pruning
- requested source path presence checks
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
- Keep the strongest result and prune trailing results that are far below the top score.
- Preserve the original user query in responses even when enriched search text is used internally.
- If a query names a source path, retrieved context must include that path to be considered sufficient.
"""

from collections.abc import Sequence
from pathlib import Path

from code_context.index_store import DEFAULT_INDEX_FILENAME, JsonIndexStore
from code_context.models import IndexSnapshot, RetrievalResponse
from code_context.query_analysis import build_retrieval_query, extract_query_paths
from code_context.vector_store import search_chunks

DEFAULT_RETRIEVAL_MIN_SCORE = 0.001
DEFAULT_MINIMUM_TOP_SCORE = 0.05
DEFAULT_RELATIVE_RESULT_SCORE_FLOOR = 0.10


def load_and_retrieve_context(
    *,
    index_dir: Path | str,
    query: str,
    related_terms: Sequence[str] | None = None,
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
        related_terms=related_terms,
        limit=limit,
        min_score=min_score,
        minimum_results=minimum_results,
        minimum_top_score=minimum_top_score,
    )


def retrieve_grounded_context(
    *,
    snapshot: IndexSnapshot,
    query: str,
    related_terms: Sequence[str] | None = None,
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
    normalized_related_terms = normalize_related_terms(related_terms)

    if not normalized_query:
        return RetrievalResponse(
            query=normalized_query,
            related_terms=normalized_related_terms,
            retrieval_query=None,
            is_sufficient=False,
            insufficient_reason="Query is empty.",
            results=[],
        )

    if not snapshot.chunks:
        return RetrievalResponse(
            query=normalized_query,
            related_terms=normalized_related_terms,
            retrieval_query=None,
            is_sufficient=False,
            insufficient_reason="The index does not contain any chunks to search.",
            results=[],
        )

    retrieval_query = build_retrieval_query(
        build_query_with_related_terms(normalized_query, normalized_related_terms)
    )
    results = search_chunks(
        snapshot.chunks,
        retrieval_query,
        limit=limit,
        min_score=min_score,
    )
    results = _prune_weak_trailing_results(results)

    requested_paths = extract_query_paths(normalized_query)
    insufficient_reason = _determine_insufficient_reason(
        results=results,
        minimum_results=minimum_results,
        minimum_top_score=minimum_top_score,
        requested_paths=requested_paths,
    )

    return RetrievalResponse(
        query=normalized_query,
        related_terms=normalized_related_terms,
        retrieval_query=retrieval_query,
        is_sufficient=insufficient_reason is None,
        insufficient_reason=insufficient_reason,
        results=results,
    )


def normalize_related_terms(related_terms: Sequence[str] | None = None) -> list[str]:
    """Normalize user-provided related terms for deterministic retrieval."""
    normalized_terms: list[str] = []
    seen: set[str] = set()

    for raw_term in related_terms or []:
        term = " ".join(str(raw_term).strip().split())
        if not term:
            continue

        term_key = term.casefold()
        if term_key in seen:
            continue

        seen.add(term_key)
        normalized_terms.append(term)

    return normalized_terms


def build_query_with_related_terms(query: str, related_terms: Sequence[str] | None = None) -> str:
    """Append explicit user-provided related terms to retrieval text."""
    normalized_terms = normalize_related_terms(related_terms)
    if not normalized_terms:
        return query

    return "\n\n".join(
        [
            query,
            f"Related terms: {' '.join(normalized_terms)}",
        ]
    )



def _prune_weak_trailing_results(
    results: list,
    *,
    relative_score_floor: float = DEFAULT_RELATIVE_RESULT_SCORE_FLOOR,
) -> list:
    if not results:
        return results

    top_score = results[0].score
    if top_score <= 0:
        return results

    minimum_relative_score = top_score * relative_score_floor

    return [
        result
        for index, result in enumerate(results)
        if index == 0 or result.score >= minimum_relative_score
    ]


def _determine_insufficient_reason(
    *,
    results: list,
    minimum_results: int,
    minimum_top_score: float,
    requested_paths: frozenset[str] | None = None,
) -> str | None:
    if len(results) < minimum_results:
        return "Not enough relevant indexed context was found."

    top_score = results[0].score if results else 0.0
    if top_score < minimum_top_score:
        return "The best retrieved context was below the sufficiency score threshold."

    missing_requested_paths = _missing_requested_paths(requested_paths or frozenset(), results)
    if missing_requested_paths:
        formatted_paths = ", ".join(sorted(missing_requested_paths))
        return f"The query asked about {formatted_paths}, but retrieved context did not include that file."

    return None


def _missing_requested_paths(requested_paths: frozenset[str], results: list) -> frozenset[str]:
    if not requested_paths:
        return frozenset()

    result_paths = frozenset(_normalize_path(result.chunk.relative_path) for result in results)

    missing_paths: set[str] = set()
    for requested_path in requested_paths:
        if not any(
            requested_path == result_path
            or requested_path.endswith(f"/{result_path}")
            or result_path.endswith(f"/{requested_path}")
            for result_path in result_paths
        ):
            missing_paths.add(requested_path)

    return frozenset(missing_paths)


def _normalize_path(path: str) -> str:
    return path.strip().strip("'\"`.,:;()[]{}").replace("\\", "/").lstrip("./").lower()


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