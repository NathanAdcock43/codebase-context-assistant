from __future__ import annotations

"""
ROLE: Verify local vector-style retrieval over source chunks.
LAYER: tests
FLOW: vector_store_validation

INPUTS:
- sample SourceChunk records
- developer query text
- search limit settings
- score threshold settings

OUTPUTS:
- vector store behavior assertions

UPSTREAM:
- vector store implementation
- SourceChunk model
- SearchResult model

DOWNSTREAM:
- local test runs
- future CI guardrails
- retriever behavior
- system map review

OWNS:
- local vector search tests
- ranking behavior tests
- tokenization tests
- embedding stability tests
- score threshold tests
- validation tests

DOES_NOT_OWN:
- scanner tests
- chunker tests
- metadata store tests
- drift detection tests
- API tests
- agent workflow tests

SIDE_EFFECTS:
- none

STATE:
  reads:
    - in-memory test chunks
  writes:
    - none

NOTES:
- These tests give us a retrieval baseline before adding ChromaDB.
- The retrieval behavior should remain deterministic across runs.
"""

import pytest

from code_context.models import SourceChunk
from code_context.vector_store import (
    LocalVectorStore,
    cosine_similarity,
    embed_text,
    search_chunks,
    tokenize,
)


def test_local_vector_store_returns_best_matching_chunk_first() -> None:
    store = LocalVectorStore()
    hash_chunk = _chunk(
        chunk_id="src/scanner.py:1-5:hash",
        relative_path="src/scanner.py",
        content="def calculate_file_hash(path):\n    return hashlib.sha256(data).hexdigest()\n",
    )
    chunker_chunk = _chunk(
        chunk_id="src/chunker.py:1-5:chunk",
        relative_path="src/chunker.py",
        content="def chunk_text(content):\n    return content.splitlines()\n",
    )

    store.add_chunks([chunker_chunk, hash_chunk])

    results = store.search("calculate file hash", limit=2)

    assert [result.chunk.relative_path for result in results] == [
        "src/scanner.py",
        "src/chunker.py",
    ]
    assert results[0].score > results[1].score


def test_search_chunks_helper_builds_store_and_searches() -> None:
    chunks = [
        _chunk(
            chunk_id="src/drift.py:1-5:drift",
            relative_path="src/drift.py",
            content="detect stale indexed metadata by comparing content hashes",
        ),
        _chunk(
            chunk_id="src/api.py:1-5:api",
            relative_path="src/api.py",
            content="define FastAPI routes and response models",
        ),
    ]

    results = search_chunks(chunks, "stale hash drift", limit=1)

    assert len(results) == 1
    assert results[0].chunk.relative_path == "src/drift.py"
    assert results[0].score > 0.0


def test_local_vector_store_respects_limit() -> None:
    store = LocalVectorStore()
    store.add_chunks(
        [
            _chunk(chunk_id="a", relative_path="a.py", content="hash value one"),
            _chunk(chunk_id="b", relative_path="b.py", content="hash value two"),
            _chunk(chunk_id="c", relative_path="c.py", content="hash value three"),
        ]
    )

    results = store.search("hash value", limit=2)

    assert len(results) == 2


def test_local_vector_store_respects_min_score() -> None:
    store = LocalVectorStore()
    store.add_chunks(
        [
            _chunk(chunk_id="a", relative_path="a.py", content="hash hash hash"),
            _chunk(chunk_id="b", relative_path="b.py", content="unrelated words only"),
        ]
    )

    results = store.search("hash", min_score=0.5)

    assert [result.chunk.relative_path for result in results] == ["a.py"]


def test_local_vector_store_returns_empty_for_empty_query() -> None:
    store = LocalVectorStore()
    store.add_chunks([_chunk(chunk_id="a", relative_path="a.py", content="hash value")])

    assert store.search("") == []


def test_local_vector_store_clear_removes_chunks() -> None:
    store = LocalVectorStore()
    store.add_chunks([_chunk(chunk_id="a", relative_path="a.py", content="hash value")])

    assert store.search("hash") != []

    store.clear()

    assert store.search("hash") == []


def test_tokenize_includes_snake_case_parts() -> None:
    tokens = tokenize("calculate_file_hash")

    assert "calculate_file_hash" in tokens
    assert "calculate" in tokens
    assert "file" in tokens
    assert "hash" in tokens


def test_embed_text_is_deterministic() -> None:
    first = embed_text("calculate file hash")
    second = embed_text("calculate file hash")

    assert first == second


def test_cosine_similarity_rejects_mismatched_vector_lengths() -> None:
    with pytest.raises(ValueError, match="vectors must have the same length"):
        cosine_similarity([1.0], [1.0, 0.0])


@pytest.mark.parametrize("dimension", [0, 1, 7])
def test_vector_dimension_must_be_at_least_eight(dimension: int) -> None:
    with pytest.raises(ValueError, match="dimension must be at least 8"):
        LocalVectorStore(dimension=dimension)

    with pytest.raises(ValueError, match="dimension must be at least 8"):
        embed_text("query", dimension=dimension)


def test_search_limit_must_be_at_least_one() -> None:
    store = LocalVectorStore()
    store.add_chunks([_chunk(chunk_id="a", relative_path="a.py", content="hash value")])

    with pytest.raises(ValueError, match="limit must be at least 1"):
        store.search("hash", limit=0)


def _chunk(*, chunk_id: str, relative_path: str, content: str) -> SourceChunk:
    return SourceChunk(
        chunk_id=chunk_id,
        relative_path=relative_path,
        start_line=1,
        end_line=max(1, content.count("\n")),
        content=content,
        language="python",
    )