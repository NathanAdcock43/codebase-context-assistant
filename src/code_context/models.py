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

OUTPUTS:
- FileMetadata
- SourceChunk

UPSTREAM:
- repository scanner
- source chunker
- future parser modules
- future index loading workflows

DOWNSTREAM:
- scanner results
- chunker results
- metadata persistence
- drift detection
- vector indexing
- API response models
- grounded answer generation

OWNS:
- shared DTO definitions
- file metadata shape
- source chunk shape
- stable field names used between project modules

DOES_NOT_OWN:
- file system traversal
- hash calculation
- source parsing
- chunk generation
- persistence writes
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