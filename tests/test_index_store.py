from __future__ import annotations

"""
ROLE: Verify JSON index persistence for file metadata, source chunks, and index snapshots.
LAYER: tests
FLOW: index_store_validation

INPUTS:
- temporary index directories
- sample FileMetadata records
- sample SourceChunk records
- sample IndexSnapshot records

OUTPUTS:
- index store behavior assertions

UPSTREAM:
- index store implementation
- FileMetadata model
- SourceChunk model
- IndexSnapshot model

DOWNSTREAM:
- local test runs
- future CI guardrails
- drift detection baseline
- system map review

OWNS:
- JSON index store tests
- snapshot round-trip tests
- missing index behavior tests
- index existence tests
- delete behavior tests

DOES_NOT_OWN:
- scanner tests
- chunker tests
- drift detection tests
- vector store tests
- API tests
- agent workflow tests

SIDE_EFFECTS:
- writes temporary JSON index files through pytest tmp_path
- deletes temporary JSON index files through pytest tmp_path

STATE:
  reads:
    - temporary JSON index files
  writes:
    - temporary JSON index files

NOTES:
- These tests keep persistence simple before drift detection.
- The stored snapshot becomes the comparison baseline for the next slice.
"""

from pathlib import Path

import pytest

from code_context.index_store import (
    DEFAULT_INDEX_FILENAME,
    JsonIndexStore,
    build_index_snapshot,
    load_index_snapshot,
    save_index_snapshot,
)
from code_context.models import FileMetadata, SourceChunk


def test_build_index_snapshot_collects_files_and_chunks(tmp_path: Path) -> None:
    file_metadata = _sample_file_metadata(tmp_path)
    source_chunk = _sample_source_chunk()

    snapshot = build_index_snapshot(
        repo_path=tmp_path,
        files=[file_metadata],
        chunks=[source_chunk],
        indexed_at=123.45,
    )

    assert snapshot.schema_version == 1
    assert snapshot.repo_root == str(tmp_path.resolve())
    assert snapshot.indexed_at == 123.45
    assert snapshot.files == [file_metadata]
    assert snapshot.chunks == [source_chunk]


def test_json_index_store_saves_and_loads_snapshot(tmp_path: Path) -> None:
    index_dir = tmp_path / ".code_context_index"
    file_metadata = _sample_file_metadata(tmp_path)
    source_chunk = _sample_source_chunk()
    snapshot = build_index_snapshot(
        repo_path=tmp_path,
        files=[file_metadata],
        chunks=[source_chunk],
        indexed_at=123.45,
    )

    store = JsonIndexStore(index_dir)

    saved_path = store.save(snapshot)
    loaded = store.load()

    assert saved_path == index_dir / DEFAULT_INDEX_FILENAME
    assert saved_path.exists()
    assert loaded == snapshot
    assert loaded.files[0].relative_path == "src/example.py"
    assert loaded.files[0].content_hash == "abc123"
    assert loaded.chunks[0].relative_path == "src/example.py"
    assert loaded.chunks[0].start_line == 1
    assert loaded.chunks[0].end_line == 2


def test_json_index_store_exists_returns_false_then_true(tmp_path: Path) -> None:
    store = JsonIndexStore(tmp_path / ".code_context_index")
    snapshot = build_index_snapshot(
        repo_path=tmp_path,
        files=[_sample_file_metadata(tmp_path)],
        indexed_at=123.45,
    )

    assert store.exists() is False

    store.save(snapshot)

    assert store.exists() is True


def test_json_index_store_load_missing_file_raises(tmp_path: Path) -> None:
    store = JsonIndexStore(tmp_path / ".code_context_index")

    with pytest.raises(FileNotFoundError, match="Index file does not exist"):
        store.load()


def test_json_index_store_delete_removes_index_file(tmp_path: Path) -> None:
    store = JsonIndexStore(tmp_path / ".code_context_index")
    snapshot = build_index_snapshot(
        repo_path=tmp_path,
        files=[_sample_file_metadata(tmp_path)],
        indexed_at=123.45,
    )

    store.save(snapshot)
    assert store.exists() is True

    store.delete()

    assert store.exists() is False


def test_save_and_load_index_snapshot_helpers_round_trip(tmp_path: Path) -> None:
    index_dir = tmp_path / ".code_context_index"
    snapshot = build_index_snapshot(
        repo_path=tmp_path,
        files=[_sample_file_metadata(tmp_path)],
        chunks=[_sample_source_chunk()],
        indexed_at=123.45,
    )

    save_index_snapshot(index_dir, snapshot)
    loaded = load_index_snapshot(index_dir)

    assert loaded == snapshot


def _sample_file_metadata(tmp_path: Path) -> FileMetadata:
    source_path = tmp_path / "src" / "example.py"

    return FileMetadata(
        path=source_path,
        relative_path="src/example.py",
        extension=".py",
        size_bytes=20,
        modified_at=100.5,
        content_hash="abc123",
        language="python",
    )


def _sample_source_chunk() -> SourceChunk:
    return SourceChunk(
        chunk_id="src/example.py:1-2:abc123",
        relative_path="src/example.py",
        start_line=1,
        end_line=2,
        content="line 1\nline 2\n",
        language="python",
    )