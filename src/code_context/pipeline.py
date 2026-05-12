from __future__ import annotations

"""
ROLE: Coordinate repository scanning, source chunking, and JSON index snapshot persistence.
LAYER: indexing
FLOW: repository_indexing_pipeline

INPUTS:
- local repository path
- local index directory
- scanner settings
- chunking settings
- index filename

OUTPUTS:
- IndexSnapshot records
- persisted JSON index snapshot

UPSTREAM:
- future CLI index command
- future FastAPI index endpoint
- local development workflow

DOWNSTREAM:
- index store
- drift detector
- vector store
- future retriever
- future API responses

OWNS:
- indexing pipeline orchestration
- scan to chunk coordination
- snapshot construction
- snapshot persistence
- repository-level indexing summary behavior

DOES_NOT_OWN:
- repository traversal rules
- file hash calculation
- chunk boundary logic
- JSON serialization details
- drift comparison
- vector scoring
- LLM prompting
- API routing

SIDE_EFFECTS:
- reads repository files through scanner and chunker
- writes JSON index snapshot through index store

STATE:
  reads:
    - local repository files
  writes:
    - local JSON index file

NOTES:
- This module wires existing deterministic pieces together.
- Keep it small so FastAPI and CLI can call the same indexing behavior later.
- Do not add AI behavior here.
"""

from pathlib import Path
from typing import Iterable

from code_context.chunker import chunk_file
from code_context.index_store import DEFAULT_INDEX_FILENAME, JsonIndexStore, build_index_snapshot
from code_context.models import FileMetadata, IndexSnapshot, SourceChunk
from code_context.scanner import (
    DEFAULT_EXCLUDED_DIRS,
    DEFAULT_SUPPORTED_EXTENSIONS,
    scan_repository,
)


def index_repository(
    repo_path: Path | str,
    *,
    index_dir: Path | str,
    supported_extensions: Iterable[str] = DEFAULT_SUPPORTED_EXTENSIONS,
    excluded_dirs: Iterable[str] = DEFAULT_EXCLUDED_DIRS,
    max_lines: int = 80,
    overlap_lines: int = 10,
    index_filename: str = DEFAULT_INDEX_FILENAME,
) -> IndexSnapshot:
    """Scan, chunk, build, save, and return an index snapshot."""
    snapshot = build_repository_snapshot(
        repo_path,
        supported_extensions=supported_extensions,
        excluded_dirs=excluded_dirs,
        max_lines=max_lines,
        overlap_lines=overlap_lines,
    )

    JsonIndexStore(index_dir, index_filename=index_filename).save(snapshot)

    return snapshot


def build_repository_snapshot(
    repo_path: Path | str,
    *,
    supported_extensions: Iterable[str] = DEFAULT_SUPPORTED_EXTENSIONS,
    excluded_dirs: Iterable[str] = DEFAULT_EXCLUDED_DIRS,
    max_lines: int = 80,
    overlap_lines: int = 10,
) -> IndexSnapshot:
    """Scan and chunk a repository without saving the snapshot."""
    files = scan_repository(
        repo_path,
        supported_extensions=supported_extensions,
        excluded_dirs=excluded_dirs,
    )
    chunks = chunk_scanned_files(
        files,
        max_lines=max_lines,
        overlap_lines=overlap_lines,
    )

    return build_index_snapshot(
        repo_path=repo_path,
        files=files,
        chunks=chunks,
    )


def chunk_scanned_files(
    files: Iterable[FileMetadata],
    *,
    max_lines: int = 80,
    overlap_lines: int = 10,
) -> list[SourceChunk]:
    """Chunk all scanned files using their stored paths, relative paths, and language labels."""
    chunks: list[SourceChunk] = []

    for file_metadata in files:
        chunks.extend(
            chunk_file(
                file_metadata.path,
                relative_path=file_metadata.relative_path,
                language=file_metadata.language,
                max_lines=max_lines,
                overlap_lines=overlap_lines,
            )
        )

    return chunks