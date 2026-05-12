from __future__ import annotations

"""
ROLE: Compare indexed file metadata against current file metadata and report stale context.
LAYER: drift
FLOW: drift_detection

INPUTS:
- indexed FileMetadata records
- current FileMetadata records
- IndexSnapshot records

OUTPUTS:
- DriftReport records
- FileDrift records grouped by added, modified, removed, and unchanged files

UPSTREAM:
- index store
- repository scanner
- future FastAPI drift endpoint
- future indexing command

DOWNSTREAM:
- API response models
- verifier
- responder
- developer-facing stale context warnings
- future reindex decision logic

OWNS:
- file metadata comparison by relative path
- added file detection
- removed file detection
- modified file detection using content hashes
- unchanged file reporting
- stale index determination

DOES_NOT_OWN:
- repository traversal
- file hashing
- source chunking
- index persistence
- vector storage
- semantic retrieval
- LLM prompting
- API routing

SIDE_EFFECTS:
- none

STATE:
  reads:
    - indexed metadata passed by caller
    - current metadata passed by caller
  writes:
    - none

NOTES:
- Content hash differences are treated as stale context.
- Modified timestamps are included in results for inspection, but timestamp-only changes do not make content stale.
- Relative paths are normalized to forward slashes before comparison.
"""

from collections.abc import Iterable

from code_context.models import DriftReport, FileDrift, FileMetadata, IndexSnapshot


def detect_drift(
    *,
    snapshot: IndexSnapshot,
    current_files: Iterable[FileMetadata],
) -> DriftReport:
    """Compare an index snapshot to current scanned files."""
    return compare_file_metadata(
        indexed_files=snapshot.files,
        current_files=current_files,
    )


def compare_file_metadata(
    *,
    indexed_files: Iterable[FileMetadata],
    current_files: Iterable[FileMetadata],
) -> DriftReport:
    """Compare indexed file metadata to current file metadata."""
    indexed_by_path = _metadata_by_relative_path(indexed_files)
    current_by_path = _metadata_by_relative_path(current_files)

    indexed_paths = set(indexed_by_path)
    current_paths = set(current_by_path)

    added = [
        _build_added_drift(current_by_path[path])
        for path in sorted(current_paths - indexed_paths)
    ]

    removed = [
        _build_removed_drift(indexed_by_path[path])
        for path in sorted(indexed_paths - current_paths)
    ]

    modified: list[FileDrift] = []
    unchanged: list[FileDrift] = []

    for path in sorted(indexed_paths & current_paths):
        indexed = indexed_by_path[path]
        current = current_by_path[path]

        if indexed.content_hash != current.content_hash:
            modified.append(_build_modified_drift(indexed=indexed, current=current))
        else:
            unchanged.append(_build_unchanged_drift(indexed=indexed, current=current))

    return DriftReport(
        is_stale=bool(added or modified or removed),
        added=added,
        modified=modified,
        removed=removed,
        unchanged=unchanged,
    )


def _metadata_by_relative_path(files: Iterable[FileMetadata]) -> dict[str, FileMetadata]:
    by_path: dict[str, FileMetadata] = {}

    for file_metadata in files:
        normalized_path = _normalize_relative_path(file_metadata.relative_path)

        if normalized_path in by_path:
            raise ValueError(f"Duplicate relative path in file metadata: {normalized_path}")

        by_path[normalized_path] = file_metadata.model_copy(
            update={"relative_path": normalized_path}
        )

    return by_path


def _build_added_drift(current: FileMetadata) -> FileDrift:
    return FileDrift(
        relative_path=current.relative_path,
        status="added",
        current_hash=current.content_hash,
        current_modified_at=current.modified_at,
        current_size_bytes=current.size_bytes,
    )


def _build_removed_drift(indexed: FileMetadata) -> FileDrift:
    return FileDrift(
        relative_path=indexed.relative_path,
        status="removed",
        indexed_hash=indexed.content_hash,
        indexed_modified_at=indexed.modified_at,
        indexed_size_bytes=indexed.size_bytes,
    )


def _build_modified_drift(*, indexed: FileMetadata, current: FileMetadata) -> FileDrift:
    return FileDrift(
        relative_path=current.relative_path,
        status="modified",
        indexed_hash=indexed.content_hash,
        current_hash=current.content_hash,
        indexed_modified_at=indexed.modified_at,
        current_modified_at=current.modified_at,
        indexed_size_bytes=indexed.size_bytes,
        current_size_bytes=current.size_bytes,
    )


def _build_unchanged_drift(*, indexed: FileMetadata, current: FileMetadata) -> FileDrift:
    return FileDrift(
        relative_path=current.relative_path,
        status="unchanged",
        indexed_hash=indexed.content_hash,
        current_hash=current.content_hash,
        indexed_modified_at=indexed.modified_at,
        current_modified_at=current.modified_at,
        indexed_size_bytes=indexed.size_bytes,
        current_size_bytes=current.size_bytes,
    )


def _normalize_relative_path(relative_path: str) -> str:
    return relative_path.replace("\\", "/")