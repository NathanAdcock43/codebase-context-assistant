from __future__ import annotations

"""
ROLE: Verify drift detection between indexed file metadata and current scanned file metadata.
LAYER: tests
FLOW: drift_validation

INPUTS:
- indexed FileMetadata records
- current FileMetadata records
- IndexSnapshot records

OUTPUTS:
- drift detection behavior assertions

UPSTREAM:
- drift implementation
- FileMetadata model
- IndexSnapshot model

DOWNSTREAM:
- local test runs
- future CI guardrails
- stale context API behavior
- system map review

OWNS:
- added file drift tests
- removed file drift tests
- modified file drift tests
- unchanged file tests
- stale report tests
- duplicate relative path tests

DOES_NOT_OWN:
- scanner tests
- chunker tests
- metadata store tests
- vector store tests
- API tests
- agent workflow tests

SIDE_EFFECTS:
- none

STATE:
  reads:
    - in-memory test metadata
  writes:
    - none

NOTES:
- These tests protect the stale-context behavior needed before vector search or agent answers.
- Timestamp-only changes are reported but do not make the index stale when content hashes match.
"""

from pathlib import Path

import pytest

from code_context.drift import compare_file_metadata, detect_drift
from code_context.models import FileMetadata, IndexSnapshot


def test_compare_file_metadata_reports_unchanged_files() -> None:
    indexed = [_file_metadata("src/app.py", content_hash="same", modified_at=1.0)]
    current = [_file_metadata("src/app.py", content_hash="same", modified_at=2.0)]

    report = compare_file_metadata(indexed_files=indexed, current_files=current)

    assert report.is_stale is False
    assert report.added == []
    assert report.modified == []
    assert report.removed == []
    assert len(report.unchanged) == 1
    assert report.unchanged[0].relative_path == "src/app.py"
    assert report.unchanged[0].status == "unchanged"
    assert report.unchanged[0].indexed_modified_at == 1.0
    assert report.unchanged[0].current_modified_at == 2.0


def test_compare_file_metadata_reports_added_files() -> None:
    report = compare_file_metadata(
        indexed_files=[],
        current_files=[_file_metadata("src/new_file.py", content_hash="new")],
    )

    assert report.is_stale is True
    assert [item.relative_path for item in report.added] == ["src/new_file.py"]
    assert report.added[0].status == "added"
    assert report.added[0].indexed_hash is None
    assert report.added[0].current_hash == "new"


def test_compare_file_metadata_reports_removed_files() -> None:
    report = compare_file_metadata(
        indexed_files=[_file_metadata("src/old_file.py", content_hash="old")],
        current_files=[],
    )

    assert report.is_stale is True
    assert [item.relative_path for item in report.removed] == ["src/old_file.py"]
    assert report.removed[0].status == "removed"
    assert report.removed[0].indexed_hash == "old"
    assert report.removed[0].current_hash is None


def test_compare_file_metadata_reports_modified_files() -> None:
    indexed = [_file_metadata("src/app.py", content_hash="old", size_bytes=10)]
    current = [_file_metadata("src/app.py", content_hash="new", size_bytes=20)]

    report = compare_file_metadata(indexed_files=indexed, current_files=current)

    assert report.is_stale is True
    assert [item.relative_path for item in report.modified] == ["src/app.py"]
    assert report.modified[0].status == "modified"
    assert report.modified[0].indexed_hash == "old"
    assert report.modified[0].current_hash == "new"
    assert report.modified[0].indexed_size_bytes == 10
    assert report.modified[0].current_size_bytes == 20


def test_compare_file_metadata_reports_mixed_drift_in_sorted_order() -> None:
    indexed = [
        _file_metadata("src/removed.py", content_hash="removed"),
        _file_metadata("src/modified.py", content_hash="old"),
        _file_metadata("src/unchanged.py", content_hash="same"),
    ]
    current = [
        _file_metadata("src/unchanged.py", content_hash="same"),
        _file_metadata("src/added.py", content_hash="added"),
        _file_metadata("src/modified.py", content_hash="new"),
    ]

    report = compare_file_metadata(indexed_files=indexed, current_files=current)

    assert report.is_stale is True
    assert [item.relative_path for item in report.added] == ["src/added.py"]
    assert [item.relative_path for item in report.modified] == ["src/modified.py"]
    assert [item.relative_path for item in report.removed] == ["src/removed.py"]
    assert [item.relative_path for item in report.unchanged] == ["src/unchanged.py"]


def test_detect_drift_uses_snapshot_files() -> None:
    snapshot = IndexSnapshot(
        repo_root="C:/example/repo",
        indexed_at=123.45,
        files=[_file_metadata("src/app.py", content_hash="old")],
        chunks=[],
    )
    current_files = [_file_metadata("src/app.py", content_hash="new")]

    report = detect_drift(snapshot=snapshot, current_files=current_files)

    assert report.is_stale is True
    assert [item.relative_path for item in report.modified] == ["src/app.py"]


def test_compare_file_metadata_normalizes_windows_style_relative_paths() -> None:
    indexed = [_file_metadata("src\\app.py", content_hash="same")]
    current = [_file_metadata("src/app.py", content_hash="same")]

    report = compare_file_metadata(indexed_files=indexed, current_files=current)

    assert report.is_stale is False
    assert [item.relative_path for item in report.unchanged] == ["src/app.py"]


def test_compare_file_metadata_rejects_duplicate_relative_paths() -> None:
    indexed = [
        _file_metadata("src/app.py", content_hash="one"),
        _file_metadata("src\\app.py", content_hash="two"),
    ]

    with pytest.raises(ValueError, match="Duplicate relative path"):
        compare_file_metadata(indexed_files=indexed, current_files=[])


def _file_metadata(
    relative_path: str,
    *,
    content_hash: str,
    modified_at: float = 1.0,
    size_bytes: int = 10,
) -> FileMetadata:
    return FileMetadata(
        path=Path(relative_path),
        relative_path=relative_path,
        extension=Path(relative_path).suffix,
        size_bytes=size_bytes,
        modified_at=modified_at,
        content_hash=content_hash,
        language="python",
    )