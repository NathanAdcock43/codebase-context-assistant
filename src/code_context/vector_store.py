from __future__ import annotations

"""
ROLE: Provide deterministic local vector-style retrieval over source chunks.
LAYER: retrieval
FLOW: local_vector_search

INPUTS:
- SourceChunk records
- developer query text and path-like query fragments
- embedding dimension
- result limit
- minimum score threshold
- chunk file paths for implementation-location matching
- implementation-location query intent signals

OUTPUTS:
- SearchResult records ordered by descending relevance score
- boosted ranking for exact path, path suffix, file name, symbol, implementation-source, and source-content token matches

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
- exact path, path suffix, and symbol match boosting for local retrieval quality
- implementation-source preference when the query asks where behavior is implemented

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
- Exact path, path suffix, and symbol boosting help implementation-location questions without hiding source references.
- Implementation questions should prefer source files over tests unless the query is explicitly about tests.
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
PATH_LIKE_PATTERN = re.compile(r"(?:[A-Za-z0-9_.-]+[\\/])+[A-Za-z0-9_.-]+\.[A-Za-z0-9_]+")

EXACT_PATH_MATCH_BOOST = 4.0
PATH_SUFFIX_MATCH_BOOST = 2.4
GENERATED_METADATA_PATH_PENALTY = 2.0
PATH_MATCH_BOOST_PER_TOKEN = 0.6
PATH_MATCH_BOOST_CAP = 1.2
CONTENT_MATCH_BOOST_PER_TOKEN = 0.02
CONTENT_MATCH_BOOST_CAP = 0.18
IMPLEMENTATION_SOURCE_BOOST = 0.7
TEST_FILE_IMPLEMENTATION_PENALTY = 0.35
IMPLEMENTATION_QUERY_MARKERS = frozenset(
    {
        "defined",
        "handled",
        "implement",
        "implemented",
        "implementation",
        "located",
        "where",
        "wired",
        "wire",
        "wiring",
    }
)
TEST_QUERY_MARKERS = frozenset(
    {
        "coverage",
        "test",
        "tested",
        "testing",
        "tests",
        "verify",
        "verifies",
    }
)


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
        query_paths = _extract_query_paths(query)
        query_vector = embed_text(query, dimension=self.dimension)
        if not any(query_vector):
            return []

        results: list[SearchResult] = []

        for entry in self._entries:
            score = _score_entry(
                query_vector=query_vector,
                query_tokens=query_tokens,
                query_paths=query_paths,
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
    query_paths: frozenset[str],
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
    exact_path_boost = _path_match_boost(query_paths, entry.chunk.relative_path)
    implementation_boost = _implementation_source_boost(query_tokens, entry.chunk)
    metadata_penalty = _generated_metadata_path_penalty(query_paths, entry.chunk.relative_path)

    return vector_score + path_boost + content_boost + exact_path_boost + implementation_boost + metadata_penalty


def _extract_query_paths(query: str) -> frozenset[str]:
    return frozenset(_normalize_path(match) for match in PATH_LIKE_PATTERN.findall(query))


def _path_match_boost(query_paths: frozenset[str], relative_path: str) -> float:
    if not query_paths:
        return 0.0

    normalized_path = _normalize_path(relative_path)
    file_name = normalized_path.rsplit("/", maxsplit=1)[-1]

    for query_path in query_paths:
        query_file_name = query_path.rsplit("/", maxsplit=1)[-1]

        if (
            query_path == normalized_path
            or query_path.endswith(f"/{normalized_path}")
            or normalized_path.endswith(f"/{query_path}")
        ):
            return EXACT_PATH_MATCH_BOOST

        if query_path.endswith(normalized_path) or normalized_path.endswith(query_path):
            return PATH_SUFFIX_MATCH_BOOST

        if query_file_name and query_file_name == file_name:
            return PATH_SUFFIX_MATCH_BOOST

    return 0.0


def _generated_metadata_path_penalty(query_paths: frozenset[str], relative_path: str) -> float:
    if not query_paths:
        return 0.0

    normalized_path = _normalize_path(relative_path)

    if ".egg-info/" in normalized_path or normalized_path.endswith("sources.txt"):
        return -GENERATED_METADATA_PATH_PENALTY

    return 0.0


def _normalize_path(path: str) -> str:
    return path.strip().strip("'\"`.,:;()[]{}").replace("\\", "/").lstrip("./").lower()


def _implementation_source_boost(query_tokens: frozenset[str], chunk: SourceChunk) -> float:
    if not _looks_like_implementation_query(query_tokens):
        return 0.0

    if _is_test_path(chunk.relative_path):
        return -TEST_FILE_IMPLEMENTATION_PENALTY

    if _is_source_path(chunk.relative_path):
        return IMPLEMENTATION_SOURCE_BOOST

    return 0.0


def _looks_like_implementation_query(query_tokens: frozenset[str]) -> bool:
    return bool(query_tokens & IMPLEMENTATION_QUERY_MARKERS) and not bool(
        query_tokens & TEST_QUERY_MARKERS
    )


def _is_source_path(relative_path: str) -> bool:
    normalized_path = relative_path.replace("\\", "/").lower()
    return normalized_path.startswith("src/")


def _is_test_path(relative_path: str) -> bool:
    normalized_path = relative_path.replace("\\", "/").lower()
    return normalized_path.startswith("tests/") or "/tests/" in normalized_path


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
