from __future__ import annotations

"""
ROLE: Persist and load local index snapshots containing scanned file metadata and source chunks.
LAYER: indexing
FLOW: metadata_persistence

INPUTS:
- repository path
- FileMetadata records
- SourceChunk records
- local index directory

OUTPUTS:
- persisted JSON index file
- loaded IndexSnapshot records

UPSTREAM:
- repository scanner
- source chunker
- future indexing command
- future FastAPI index endpoint

DOWNSTREAM:
- drift detector
- vector indexing
- retriever
- verifier
- API response models

OWNS:
- local JSON index snapshot persistence
- index directory creation
- index file path conventions
- IndexSnapshot serialization
- IndexSnapshot loading

DOES_NOT_OWN:
- repository traversal
- file hashing
- source chunking
- drift comparison
- vector embedding
- semantic retrieval
- LLM prompting
- API routing

SIDE_EFFECTS:
- creates index directory when saving
- writes JSON index snapshots
- reads JSON index snapshots
- deletes local index file when requested

STATE:
  reads:
    - local JSON index file
  writes:
    - local JSON index file

NOTES:
- Keep persistence JSON-based for the MVP.
- This gives drift detection a stable stored baseline before adding vector search.
- Read with utf-8-sig tolerance so a BOM does not break local recovery.
"""

import time
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel

from code_context.models import FileMetadata, IndexSnapshot, SourceChunk


DEFAULT_INDEX_FILENAME = "index.json"


class JsonIndexStore:
    """Small JSON-backed store for local index snapshots."""

    def __init__(
        self,
        index_dir: Path | str,
        *,
        index_filename: str = DEFAULT_INDEX_FILENAME,
    ) -> None:
        self.index_dir = Path(index_dir)
        self.index_filename = index_filename

    @property
    def index_path(self) -> Path:
        return self.index_dir / self.index_filename

    def exists(self) -> bool:
        return self.index_path.exists()

    def save(self, snapshot: IndexSnapshot) -> Path:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            _dump_model_json(snapshot),
            encoding="utf-8",
        )
        return self.index_path

    def load(self) -> IndexSnapshot:
        if not self.index_path.exists():
            raise FileNotFoundError(f"Index file does not exist: {self.index_path}")

        content = self.index_path.read_text(encoding="utf-8-sig")
        return _validate_model_json(IndexSnapshot, content)

    def delete(self) -> None:
        if self.index_path.exists():
            self.index_path.unlink()


def build_index_snapshot(
    *,
    repo_path: Path | str,
    files: Iterable[FileMetadata],
    chunks: Iterable[SourceChunk] | None = None,
    indexed_at: float | None = None,
) -> IndexSnapshot:
    """Build an index snapshot from scanned files and generated chunks."""
    repo_root = Path(repo_path).resolve()

    return IndexSnapshot(
        repo_root=str(repo_root),
        indexed_at=indexed_at if indexed_at is not None else time.time(),
        files=list(files),
        chunks=list(chunks or []),
    )


def save_index_snapshot(
    index_dir: Path | str,
    snapshot: IndexSnapshot,
    *,
    index_filename: str = DEFAULT_INDEX_FILENAME,
) -> Path:
    """Save an index snapshot using the default JSON store."""
    return JsonIndexStore(index_dir, index_filename=index_filename).save(snapshot)


def load_index_snapshot(
    index_dir: Path | str,
    *,
    index_filename: str = DEFAULT_INDEX_FILENAME,
) -> IndexSnapshot:
    """Load an index snapshot using the default JSON store."""
    return JsonIndexStore(index_dir, index_filename=index_filename).load()


def _dump_model_json(model: BaseModel) -> str:
    if hasattr(model, "model_dump_json"):
        return model.model_dump_json(indent=2)

    return model.json(indent=2)


def _validate_model_json(model_type: type[IndexSnapshot], content: str) -> IndexSnapshot:
    if hasattr(model_type, "model_validate_json"):
        return model_type.model_validate_json(content)

    return model_type.parse_raw(content)