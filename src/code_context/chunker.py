from __future__ import annotations

"""
ROLE: Split source text into deterministic chunks while preserving file and line references.
LAYER: chunking
FLOW: source_chunking

INPUTS:
- source file content
- repository-relative file path
- language label
- maximum lines per chunk
- overlap lines between chunks

OUTPUTS:
- SourceChunk records with stable chunk IDs and line references

UPSTREAM:
- repository scanner
- future parser
- future indexing command
- future FastAPI index endpoint

DOWNSTREAM:
- metadata index store
- vector store
- retriever
- verifier
- grounded answer generation

OWNS:
- line-based source chunking
- chunk boundary calculation
- overlap handling
- stable chunk ID generation
- source line reference preservation

DOES_NOT_OWN:
- repository traversal
- file metadata hashing
- AST-aware parsing
- metadata persistence
- vector embedding
- semantic retrieval
- LLM prompting

SIDE_EFFECTS:
- chunk_file reads source files from disk

STATE:
  reads:
    - source files when chunk_file is used
  writes:
    - none

NOTES:
- Keep this chunker line-based for the MVP.
- AST-aware or symbol-aware chunking can be added later if needed.
- The key requirement here is grounded references back to file path and line range.
"""

import hashlib
from pathlib import Path

from code_context.models import SourceChunk


def chunk_text(
    content: str,
    *,
    relative_path: str,
    language: str = "unknown",
    max_lines: int = 80,
    overlap_lines: int = 10,
) -> list[SourceChunk]:
    """Split source text into line-based chunks with preserved line references."""
    _validate_chunk_settings(max_lines=max_lines, overlap_lines=overlap_lines)

    lines = content.splitlines(keepends=True)
    if not lines:
        return []

    normalized_path = _normalize_relative_path(relative_path)
    step = max_lines - overlap_lines
    chunks: list[SourceChunk] = []

    start_index = 0
    while start_index < len(lines):
        end_index = min(start_index + max_lines, len(lines))
        start_line = start_index + 1
        end_line = end_index
        chunk_content = "".join(lines[start_index:end_index])

        chunks.append(
            SourceChunk(
                chunk_id=build_chunk_id(
                    relative_path=normalized_path,
                    start_line=start_line,
                    end_line=end_line,
                    content=chunk_content,
                ),
                relative_path=normalized_path,
                start_line=start_line,
                end_line=end_line,
                content=chunk_content,
                language=language,
            )
        )

        if end_index == len(lines):
            break

        start_index += step

    return chunks


def chunk_file(
    path: Path | str,
    *,
    relative_path: str | None = None,
    language: str = "unknown",
    max_lines: int = 80,
    overlap_lines: int = 10,
    encoding: str = "utf-8",
    errors: str = "replace",
) -> list[SourceChunk]:
    """Read a source file and split it into line-based chunks."""
    source_path = Path(path)
    chunk_relative_path = relative_path or source_path.as_posix()

    with source_path.open("r", encoding=encoding, errors=errors, newline="") as file:
        content = file.read()

    return chunk_text(
        content,
        relative_path=chunk_relative_path,
        language=language,
        max_lines=max_lines,
        overlap_lines=overlap_lines,
    )


def build_chunk_id(
    *,
    relative_path: str,
    start_line: int,
    end_line: int,
    content: str,
) -> str:
    """Build a stable chunk ID from path, line range, and chunk content."""
    normalized_path = _normalize_relative_path(relative_path)
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()[:12]
    return f"{normalized_path}:{start_line}-{end_line}:{content_hash}"


def _validate_chunk_settings(*, max_lines: int, overlap_lines: int) -> None:
    if max_lines < 1:
        raise ValueError("max_lines must be at least 1")

    if overlap_lines < 0:
        raise ValueError("overlap_lines must be 0 or greater")

    if overlap_lines >= max_lines:
        raise ValueError("overlap_lines must be less than max_lines")


def _normalize_relative_path(relative_path: str) -> str:
    return relative_path.replace("\\", "/")