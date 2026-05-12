from __future__ import annotations

"""
ROLE: Scan a local repository and return structured file metadata for supported source files.
LAYER: scanner
FLOW: repository_scan

INPUTS:
- local repository path
- supported file extension rules
- ignored directory rules
- file system metadata
- file contents for hashing

OUTPUTS:
- FileMetadata records for supported files

UPSTREAM:
- CLI indexing command
- future FastAPI index endpoint
- future drift detection workflow
- future system indexing workflow

DOWNSTREAM:
- chunker
- index store
- drift detector
- vector indexing
- system map review

OWNS:
- repository traversal
- ignored directory filtering
- supported file extension filtering
- SHA-256 content hash calculation
- basic language inference
- FileMetadata construction

DOES_NOT_OWN:
- source code chunking
- AST parsing
- metadata persistence
- vector storage
- drift comparison
- LLM prompting
- API routing

SIDE_EFFECTS:
- reads file system metadata
- reads supported source files for hashing

STATE:
  reads:
    - local repository files
  writes:
    - none

NOTES:
- Keep scanner behavior deterministic and easy to test.
- Prefer repo-relative paths for anything that may be shown to users.
- Do not add AI behavior here.
"""

import hashlib
from pathlib import Path
from typing import Iterable

from code_context.models import FileMetadata


DEFAULT_EXCLUDED_DIRS = frozenset(
    {
        ".git",
        ".idea",
        ".venv",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        ".code_context_index",
        ".chroma",
    }
)

DEFAULT_SUPPORTED_EXTENSIONS = frozenset(
    {
        ".py",
        ".md",
        ".txt",
        ".toml",
        ".json",
        ".yaml",
        ".yml",
        ".js",
        ".ts",
        ".java",
        ".sql",
    }
)

LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".md": "markdown",
    ".txt": "text",
    ".toml": "toml",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".sql": "sql",
}


def scan_repository(
    repo_path: Path | str,
    *,
    supported_extensions: Iterable[str] = DEFAULT_SUPPORTED_EXTENSIONS,
    excluded_dirs: Iterable[str] = DEFAULT_EXCLUDED_DIRS,
) -> list[FileMetadata]:
    """Scan a repository and return metadata for supported source files."""
    root = Path(repo_path).resolve()

    if not root.exists():
        raise FileNotFoundError(f"Repository path does not exist: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {root}")

    normalized_extensions = _normalize_extensions(supported_extensions)
    excluded_dir_names = set(excluded_dirs)

    metadata: list[FileMetadata] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if not should_scan_file(
            path,
            repo_root=root,
            supported_extensions=normalized_extensions,
            excluded_dirs=excluded_dir_names,
        ):
            continue

        stat = path.stat()
        relative_path = path.relative_to(root).as_posix()

        metadata.append(
            FileMetadata(
                path=path,
                relative_path=relative_path,
                extension=path.suffix.lower(),
                size_bytes=stat.st_size,
                modified_at=stat.st_mtime,
                content_hash=calculate_file_hash(path),
                language=infer_language(path),
            )
        )

    return metadata


def should_scan_file(
    path: Path,
    *,
    repo_root: Path,
    supported_extensions: Iterable[str] = DEFAULT_SUPPORTED_EXTENSIONS,
    excluded_dirs: Iterable[str] = DEFAULT_EXCLUDED_DIRS,
) -> bool:
    """Return whether a file should be included in repository scanning."""
    normalized_extensions = _normalize_extensions(supported_extensions)

    if path.suffix.lower() not in normalized_extensions:
        return False

    try:
        relative_parts = path.relative_to(repo_root).parts
    except ValueError:
        return False

    excluded_dir_names = set(excluded_dirs)
    return not any(part in excluded_dir_names for part in relative_parts)


def calculate_file_hash(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Calculate a SHA-256 hash for a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)

    return digest.hexdigest()


def infer_language(path: Path) -> str:
    """Infer a simple language label from a file extension."""
    return LANGUAGE_BY_EXTENSION.get(path.suffix.lower(), "unknown")


def _normalize_extensions(extensions: Iterable[str]) -> set[str]:
    normalized: set[str] = set()

    for extension in extensions:
        clean_extension = extension.strip().lower()
        if not clean_extension:
            continue

        if not clean_extension.startswith("."):
            clean_extension = f".{clean_extension}"

        normalized.add(clean_extension)

    return normalized