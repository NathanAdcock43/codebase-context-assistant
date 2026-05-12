from __future__ import annotations

"""
ROLE: Verify source chunking behavior, line references, overlap handling, and stable chunk IDs.
LAYER: tests
FLOW: chunker_validation

INPUTS:
- sample source text
- temporary source files
- chunk size settings
- overlap settings

OUTPUTS:
- chunker behavior assertions

UPSTREAM:
- chunker implementation
- SourceChunk model

DOWNSTREAM:
- local test runs
- future CI guardrails
- system map review

OWNS:
- line-reference chunking tests
- chunk boundary tests
- overlap behavior tests
- chunk ID stability tests
- file chunking tests

DOES_NOT_OWN:
- scanner tests
- metadata store tests
- drift detection tests
- vector store tests
- API tests
- agent workflow tests

SIDE_EFFECTS:
- writes temporary test files through pytest tmp_path

STATE:
  reads:
    - temporary test files
  writes:
    - temporary test files

NOTES:
- These tests protect the grounding behavior needed for file and line references.
- Use small examples so chunk boundaries are easy to inspect.
"""

from pathlib import Path

import pytest

from code_context.chunker import build_chunk_id, chunk_file, chunk_text


def test_chunk_text_returns_single_chunk_with_line_references() -> None:
    content = "line 1\nline 2\nline 3\n"

    chunks = chunk_text(
        content,
        relative_path="src/example.py",
        language="python",
        max_lines=10,
        overlap_lines=0,
    )

    assert len(chunks) == 1
    assert chunks[0].relative_path == "src/example.py"
    assert chunks[0].start_line == 1
    assert chunks[0].end_line == 3
    assert chunks[0].content == content
    assert chunks[0].language == "python"
    assert chunks[0].chunk_id.startswith("src/example.py:1-3:")


def test_chunk_text_splits_large_content_by_max_lines() -> None:
    content = "line 1\nline 2\nline 3\nline 4\nline 5\n"

    chunks = chunk_text(
        content,
        relative_path="src/example.py",
        max_lines=2,
        overlap_lines=0,
    )

    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [
        (1, 2),
        (3, 4),
        (5, 5),
    ]
    assert [chunk.content for chunk in chunks] == [
        "line 1\nline 2\n",
        "line 3\nline 4\n",
        "line 5\n",
    ]


def test_chunk_text_uses_overlap_lines() -> None:
    content = "line 1\nline 2\nline 3\nline 4\nline 5\n"

    chunks = chunk_text(
        content,
        relative_path="src/example.py",
        max_lines=3,
        overlap_lines=1,
    )

    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [
        (1, 3),
        (3, 5),
    ]
    assert chunks[0].content == "line 1\nline 2\nline 3\n"
    assert chunks[1].content == "line 3\nline 4\nline 5\n"


def test_chunk_text_returns_empty_list_for_empty_content() -> None:
    chunks = chunk_text(
        "",
        relative_path="src/empty.py",
        max_lines=10,
        overlap_lines=0,
    )

    assert chunks == []


@pytest.mark.parametrize(
    ("max_lines", "overlap_lines", "expected_message"),
    [
        (0, 0, "max_lines must be at least 1"),
        (5, -1, "overlap_lines must be 0 or greater"),
        (5, 5, "overlap_lines must be less than max_lines"),
        (5, 6, "overlap_lines must be less than max_lines"),
    ],
)
def test_chunk_text_rejects_invalid_chunk_settings(
    max_lines: int,
    overlap_lines: int,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        chunk_text(
            "line 1\n",
            relative_path="src/example.py",
            max_lines=max_lines,
            overlap_lines=overlap_lines,
        )


def test_chunk_file_reads_file_and_preserves_relative_path(tmp_path: Path) -> None:
    source_file = tmp_path / "example.py"
    source_file.write_bytes(b"alpha\r\nbeta\r\ngamma\r\n")

    chunks = chunk_file(
        source_file,
        relative_path="src/example.py",
        language="python",
        max_lines=2,
        overlap_lines=0,
    )

    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [
        (1, 2),
        (3, 3),
    ]
    assert chunks[0].relative_path == "src/example.py"
    assert chunks[0].content == "alpha\r\nbeta\r\n"
    assert chunks[1].content == "gamma\r\n"


def test_chunk_ids_are_stable_for_same_content() -> None:
    content = "line 1\nline 2\n"

    first_chunks = chunk_text(
        content,
        relative_path="src/example.py",
        max_lines=2,
        overlap_lines=0,
    )
    second_chunks = chunk_text(
        content,
        relative_path="src/example.py",
        max_lines=2,
        overlap_lines=0,
    )

    assert first_chunks[0].chunk_id == second_chunks[0].chunk_id


def test_chunk_ids_include_normalized_relative_path() -> None:
    chunk_id = build_chunk_id(
        relative_path="src\\example.py",
        start_line=1,
        end_line=2,
        content="line 1\nline 2\n",
    )

    assert chunk_id.startswith("src/example.py:1-2:")