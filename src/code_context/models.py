from __future__ import annotations

"""
ROLE: Define shared data models used by the codebase context assistant.
LAYER: core_model
FLOW: shared_models

INPUTS:
- repository file paths
- file system metadata
- calculated content hashes
- inferred language values
- source chunk boundaries
- source chunk content
- persisted index snapshot data
- optional offline file enrichment metadata
- drift comparison results
- retrieval scores
- retrieval sufficiency decisions

OUTPUTS:
- FileMetadata
- SourceChunk
- FileEnrichment
- IndexSnapshot
- FileDrift
- DriftReport
- SearchResult
- RetrievalResponse

UPSTREAM:
- repository scanner
- source chunker
- index store
- drift detector
- vector store
- retrieval service
- future parser modules
- future index enrichment workflows
- future index loading workflows

DOWNSTREAM:
- scanner results
- chunker results
- metadata persistence
- drift detection
- vector indexing
- API response models
- grounded retrieval service
- grounded answer generation

OWNS:
- shared DTO definitions
- file metadata shape
- source chunk shape
- optional file enrichment metadata shape
- persisted index snapshot shape
- file drift result shape
- drift report shape
- search result shape
- retrieval response shape
- stable field names used between project modules

DOES_NOT_OWN:
- file system traversal
- hash calculation
- source parsing
- chunk generation
- persistence writes
- drift comparison logic
- retrieval scoring logic
- retrieval sufficiency logic
- API routing

SIDE_EFFECTS:
- none

STATE:
  reads:
    - none
  writes:
    - none

NOTES:
- Models should stay simple and explicit.
- Prefer adding fields when a slice actually needs them instead of guessing far ahead.
"""

from pathlib import Path

from pydantic import BaseModel, Field


class FileMetadata(BaseModel):
    path: Path
    relative_path: str
    extension: str
    size_bytes: int
    modified_at: float
    content_hash: str
    language: str = Field(default="unknown")


class SourceChunk(BaseModel):
    chunk_id: str
    relative_path: str
    start_line: int
    end_line: int
    content: str
    language: str = Field(default="unknown")



class FileEnrichment(BaseModel):
    relative_path: str
    source_hash: str
    enriched_at: float
    provider: str | None = Field(default=None)
    model: str | None = Field(default=None)
    schema_version: int = Field(default=1)
    summary: str | None = Field(default=None)
    conceptual_terms: list[str] = Field(default_factory=list)
    related_user_phrases: list[str] = Field(default_factory=list)
    owned_behaviors: list[str] = Field(default_factory=list)
    important_symbols: list[str] = Field(default_factory=list)


class IndexSnapshot(BaseModel):
    schema_version: int = Field(default=1)
    repo_root: str
    indexed_at: float
    files: list[FileMetadata] = Field(default_factory=list)
    chunks: list[SourceChunk] = Field(default_factory=list)
    enrichments: list[FileEnrichment] = Field(default_factory=list)


class FileDrift(BaseModel):
    relative_path: str
    status: str
    indexed_hash: str | None = Field(default=None)
    current_hash: str | None = Field(default=None)
    indexed_modified_at: float | None = Field(default=None)
    current_modified_at: float | None = Field(default=None)
    indexed_size_bytes: int | None = Field(default=None)
    current_size_bytes: int | None = Field(default=None)


class DriftReport(BaseModel):
    is_stale: bool
    added: list[FileDrift] = Field(default_factory=list)
    modified: list[FileDrift] = Field(default_factory=list)
    removed: list[FileDrift] = Field(default_factory=list)
    unchanged: list[FileDrift] = Field(default_factory=list)


class SearchResult(BaseModel):
    chunk: SourceChunk
    score: float


class RetrievalResponse(BaseModel):
    query: str
    related_terms: list[str] = Field(default_factory=list)
    matched_related_terms: list[str] = Field(default_factory=list)
    retrieval_query: str | None = Field(default=None)
    is_sufficient: bool
    insufficient_reason: str | None = Field(default=None)
    results: list[SearchResult] = Field(default_factory=list)
