from __future__ import annotations

"""
ROLE: Provide deterministic local vector-style retrieval over source chunks.
LAYER: retrieval
FLOW: local_vector_search

INPUTS:
- SourceChunk records
- developer query text
- embedding dimension
- result limit
- minimum score threshold
- chunk file paths for implementation-location matching

OUTPUTS:
- SearchResult records ordered by descending relevance score
- boosted ranking for path, file name, symbol, and source-content token matches

UPSTREAM:
- source chunker
- index store
- FastAPI search endpoint
- FastAPI retrieve endpoint
- retriever node
- ask workflow

DOWNSTREAM:
- verifier
- responder
- grounded answer generation
- future ChromaDB-backed vector store

OWNS:
- deterministic text tokenization
- local hashed embedding generation
- cosine similarity scoring
- chunk retrieval ranking
- result limiting and score filtering
- path and symbol match boosting for local retrieval quality

DOES_NOT_OWN:
- repository traversal
- source chunking
- metadata persistence
- drift detection
- ChromaDB persistence
- LLM prompting
- answer generation
- API routing

SIDE_EFFECTS:
- none

STATE:
  reads:
    - in-memory source chunks
  writes:
    - in-memory source chunk vectors

NOTES:
- This is a dependency-free retrieval baseline for the MVP.
- It gives us testable retrieval behavior before adding ChromaDB.
- Path and symbol boosting helps implementation-location questions without hiding source references.
- Later ChromaDB storage should preserve the same public search behavior.
"""

import hashlib
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

from code_context.models import SearchResult, SourceChunk


DEFAULT_EMBEDDING_DIMENSION = 256
TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+")

PATH_MATCH_BOOST_PER_TOKEN = 0.6
PATH_MATCH_BOOST_CAP = 1.2
CONTENT_MATCH_BOOST_PER_TOKEN = 0.02
CONTENT_MATCH_BOOST_CAP = 0.18


@dataclass(frozen=True)
class _ChunkEntry:
    chunk: SourceChunk
    vector: list[float]
    path_tokens: frozenset[str]
    content_tokens: frozenset[str]


class LocalVectorStore:
    """In-memory vector-style search over source chunks."""

    def __init__(self, *, dimension: int = DEFAULT_EMBEDDING_DIMENSION) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")

        self.dimension = dimension
        self._entries: list[_ChunkEntry] = []

    def add_chunks(self, chunks: Iterable[SourceChunk]) -> None:
        """Add chunks to the local search store."""
        for chunk in chunks:
            self._entries.append(
                _ChunkEntry(
                    chunk=chunk,
                    vector=embed_text(_build_searchable_chunk_text(chunk), dimension=self.dimension),
                    path_tokens=_path_tokens(chunk),
                    content_tokens=frozenset(tokenize(chunk.content)),
                )
            )

    def clear(self) -> None:
        """Remove all indexed chunks from the local search store."""
        self._entries.clear()

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        min_score: float = 0.0,
    ) -> list[SearchResult]:
        """Search indexed chunks and return ranked results."""
        if limit < 1:
            raise ValueError("limit must be at least 1")

        query_tokens = frozenset(tokenize(query))
        query_vector = embed_text(query, dimension=self.dimension)
        if not any(query_vector):
            return []

        results: list[SearchResult] = []

        for entry in self._entries:
            score = _score_entry(
                query_vector=query_vector,
                query_tokens=query_tokens,
                entry=entry,
            )
            if score >= min_score:
                results.append(SearchResult(chunk=entry.chunk, score=score))

        results.sort(
            key=lambda result: (
                -result.score,
                result.chunk.relative_path,
                result.chunk.start_line,
                result.chunk.end_line,
            )
        )

        return results[:limit]


def search_chunks(
    chunks: Iterable[SourceChunk],
    query: str,
    *,
    limit: int = 5,
    min_score: float = 0.0,
    dimension: int = DEFAULT_EMBEDDING_DIMENSION,
) -> list[SearchResult]:
    """Convenience helper for one-off chunk searches."""
    store = LocalVectorStore(dimension=dimension)
    store.add_chunks(chunks)
    return store.search(query, limit=limit, min_score=min_score)


def embed_text(text: str, *, dimension: int = DEFAULT_EMBEDDING_DIMENSION) -> list[float]:
    """Create a deterministic hashed token vector for text."""
    if dimension < 8:
        raise ValueError("dimension must be at least 8")

    vector = [0.0] * dimension

    for token in tokenize(text):
        index = _token_index(token, dimension=dimension)
        vector[index] += 1.0

    return _normalize_vector(vector)


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Calculate cosine similarity for two same-length vectors."""
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    return dot_product / (left_norm * right_norm)


def tokenize(text: str) -> list[str]:
    """Tokenize text and include snake_case components for code search."""
    tokens: list[str] = []

    for match in TOKEN_PATTERN.findall(text):
        normalized = match.lower()
        tokens.append(normalized)

        if "_" in normalized:
            tokens.extend(part for part in normalized.split("_") if part)

    return tokens


def _score_entry(
    *,
    query_vector: list[float],
    query_tokens: frozenset[str],
    entry: _ChunkEntry,
) -> float:
    vector_score = cosine_similarity(query_vector, entry.vector)
    path_boost = _overlap_boost(
        query_tokens,
        entry.path_tokens,
        per_token=PATH_MATCH_BOOST_PER_TOKEN,
        cap=PATH_MATCH_BOOST_CAP,
    )
    content_boost = _overlap_boost(
        query_tokens,
        entry.content_tokens,
        per_token=CONTENT_MATCH_BOOST_PER_TOKEN,
        cap=CONTENT_MATCH_BOOST_CAP,
    )

    return vector_score + path_boost + content_boost


def _overlap_boost(
    query_tokens: frozenset[str],
    candidate_tokens: frozenset[str],
    *,
    per_token: float,
    cap: float,
) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0

    overlap_count = len(query_tokens & candidate_tokens)
    return min(cap, overlap_count * per_token)


def _build_searchable_chunk_text(chunk: SourceChunk) -> str:
    path_terms = " ".join(sorted(_path_tokens(chunk)))

    return "\n".join(
        [
            path_terms,
            path_terms,
            chunk.language,
            chunk.content,
        ]
    )


def _path_tokens(chunk: SourceChunk) -> frozenset[str]:
    return frozenset(
        tokenize(
            " ".join(
                [
                    chunk.relative_path,
                    chunk.language,
                ]
            )
        )
    )


def _token_index(token: str, *, dimension: int) -> int:
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") % dimension


def _normalize_vector(vector: list[float]) -> list[float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude == 0.0:
        return vector

    return [value / magnitude for value in vector]
