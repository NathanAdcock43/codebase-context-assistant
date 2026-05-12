from __future__ import annotations

"""
ROLE: Verify grounded retrieval service behavior and sufficiency checks.
LAYER: tests
FLOW: retrieval_validation

INPUTS:
- IndexSnapshot records
- sample SourceChunk records
- query text
- retrieval thresholds
- temporary index directories

OUTPUTS:
- retrieval response behavior assertions

UPSTREAM:
- retrieval service implementation
- vector store implementation
- index store implementation
- IndexSnapshot model
- SourceChunk model

DOWNSTREAM:
- local test runs
- future CI guardrails
- FastAPI retrieve endpoint
- future verifier node
- future responder node
- system map review

OWNS:
- retrieval sufficiency tests
- insufficient context tests
- empty query tests
- threshold validation tests
- loaded snapshot retrieval tests

DOES_NOT_OWN:
- scanner tests
- chunker tests
- metadata store tests
- drift detection tests
- API route tests
- LLM answer tests

SIDE_EFFECTS:
- writes temporary JSON index files through pytest tmp_path

STATE:
  reads:
    - temporary JSON index files
  writes:
    - temporary JSON index files

NOTES:
- These tests protect refusal behavior before adding LLM answer generation.
- The retrieval service returns grounded context but does not generate answers.
"""

from pathlib import Path

import pytest

from code_context.index_store import JsonIndexStore, build_index_snapshot
from code_context.models import IndexSnapshot, SourceChunk
from code_context.retrieval import load_and_retrieve_context, retrieve_grounded_context


def test_retrieve_grounded_context_marks_strong_match_as_sufficient() -> None:
    snapshot = _snapshot(
        chunks=[
            _chunk(
                chunk_id="src/scanner.py:1-2:hash",
                relative_path="src/scanner.py",
                content="def calculate_file_hash(path):\n    return hashlib.sha256(data).hexdigest()\n",
            ),
            _chunk(
                chunk_id="src/api.py:1-2:api",
                relative_path="src/api.py",
                content="define FastAPI routes for health checks\n",
            ),
        ]
    )

    response = retrieve_grounded_context(
        snapshot=snapshot,
        query="calculate file hash",
        limit=2,
        minimum_top_score=0.05,
    )

    assert response.is_sufficient is True
    assert response.insufficient_reason is None
    assert response.results[0].chunk.relative_path == "src/scanner.py"
    assert response.results[0].score > 0


def test_retrieve_grounded_context_marks_empty_query_as_insufficient() -> None:
    response = retrieve_grounded_context(
        snapshot=_snapshot(chunks=[_chunk(content="hash content")]),
        query="   ",
    )

    assert response.query == ""
    assert response.is_sufficient is False
    assert response.insufficient_reason == "Query is empty."
    assert response.results == []


def test_retrieve_grounded_context_marks_empty_index_as_insufficient() -> None:
    response = retrieve_grounded_context(
        snapshot=_snapshot(chunks=[]),
        query="hash",
    )

    assert response.query == "hash"
    assert response.is_sufficient is False
    assert response.insufficient_reason == "The index does not contain any chunks to search."
    assert response.results == []


def test_retrieve_grounded_context_marks_no_matching_results_as_insufficient() -> None:
    snapshot = _snapshot(
        chunks=[
            _chunk(
                chunk_id="src/api.py:1-2:api",
                relative_path="src/api.py",
                content="FastAPI route definitions only",
            )
        ]
    )

    response = retrieve_grounded_context(
        snapshot=snapshot,
        query="postgres liquibase migration",
        min_score=0.99,
    )

    assert response.is_sufficient is False
    assert response.insufficient_reason == "Not enough relevant indexed context was found."
    assert response.results == []


def test_retrieve_grounded_context_marks_low_score_result_as_insufficient() -> None:
    snapshot = _snapshot(
        chunks=[
            _chunk(
                chunk_id="src/scanner.py:1-2:hash",
                relative_path="src/scanner.py",
                content="hash route definitions",
            )
        ]
    )

    response = retrieve_grounded_context(
        snapshot=snapshot,
        query="hash",
        min_score=0.0,
        minimum_top_score=0.99,
    )

    assert response.is_sufficient is False
    assert (
        response.insufficient_reason
        == "The best retrieved context was below the sufficiency score threshold."
    )
    assert response.results != []


def test_load_and_retrieve_context_loads_saved_snapshot(tmp_path: Path) -> None:
    index_dir = tmp_path / ".code_context_index"
    snapshot = _snapshot(
        chunks=[
            _chunk(
                chunk_id="src/drift.py:1-2:drift",
                relative_path="src/drift.py",
                content="detect stale indexed metadata by comparing content hashes",
            )
        ]
    )

    JsonIndexStore(index_dir).save(snapshot)

    response = load_and_retrieve_context(
        index_dir=index_dir,
        query="stale content hashes",
    )

    assert response.is_sufficient is True
    assert response.results[0].chunk.relative_path == "src/drift.py"


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"limit": 0}, "limit must be at least 1"),
        ({"min_score": -0.1}, "min_score must be 0 or greater"),
        ({"minimum_results": 0}, "minimum_results must be at least 1"),
        (
            {"limit": 1, "minimum_results": 2},
            "minimum_results must be less than or equal to limit",
        ),
        ({"minimum_top_score": -0.1}, "minimum_top_score must be 0 or greater"),
    ],
)
def test_retrieve_grounded_context_rejects_invalid_settings(
    kwargs: dict,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        retrieve_grounded_context(
            snapshot=_snapshot(chunks=[_chunk(content="hash content")]),
            query="hash",
            **kwargs,
        )


def _snapshot(*, chunks: list[SourceChunk]) -> IndexSnapshot:
    return build_index_snapshot(
        repo_path="C:/example/repo",
        files=[],
        chunks=chunks,
        indexed_at=123.45,
    )


def _chunk(
    *,
    content: str,
    chunk_id: str = "src/example.py:1-1:abc123",
    relative_path: str = "src/example.py",
) -> SourceChunk:
    return SourceChunk(
        chunk_id=chunk_id,
        relative_path=relative_path,
        start_line=1,
        end_line=max(1, content.count("\n")),
        content=content,
        language="python",
    )