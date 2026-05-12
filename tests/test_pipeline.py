from __future__ import annotations

"""
ROLE: Verify repository indexing pipeline behavior from scan to chunks to saved snapshot.
LAYER: tests
FLOW: indexing_pipeline_validation

INPUTS:
- temporary repository directories
- sample source files
- ignored directory examples
- chunking settings
- temporary index directories

OUTPUTS:
- indexing pipeline behavior assertions

UPSTREAM:
- pipeline implementation
- scanner implementation
- chunker implementation
- index store implementation
- IndexSnapshot model

DOWNSTREAM:
- local test runs
- future CI guardrails
- future CLI index command
- future FastAPI index endpoint
- system map review

OWNS:
- repository indexing pipeline tests
- scan to chunk integration tests
- saved snapshot tests
- ignored directory integration tests
- custom extension integration tests

DOES_NOT_OWN:
- standalone scanner tests
- standalone chunker tests
- standalone metadata store tests
- drift detection tests
- vector store tests
- API tests
- agent workflow tests

SIDE_EFFECTS:
- writes temporary source files through pytest tmp_path
- writes temporary JSON index files through pytest tmp_path

STATE:
  reads:
    - temporary repository files
    - temporary JSON index files
  writes:
    - temporary repository files
    - temporary JSON index files

NOTES:
- These tests prove the core local indexing workflow works before adding API endpoints.
- Keep fixtures small so failures are easy to inspect.
"""

from pathlib import Path

from code_context.index_store import JsonIndexStore
from code_context.pipeline import build_repository_snapshot, chunk_scanned_files, index_repository
from code_context.scanner import scan_repository


def test_build_repository_snapshot_scans_and_chunks_supported_files(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source_dir = repo / "src"
    source_dir.mkdir(parents=True)

    app_file = source_dir / "app.py"
    app_file.write_bytes(b"line 1\nline 2\nline 3\n")

    readme_file = repo / "README.md"
    readme_file.write_bytes(b"# Example\n")

    ignored_dir = repo / ".venv"
    ignored_dir.mkdir()
    ignored_file = ignored_dir / "ignored.py"
    ignored_file.write_bytes(b"print('ignored')\n")

    snapshot = build_repository_snapshot(
        repo,
        max_lines=2,
        overlap_lines=0,
    )

    file_paths = [file.relative_path for file in snapshot.files]
    chunk_ranges = [
        (chunk.relative_path, chunk.start_line, chunk.end_line)
        for chunk in snapshot.chunks
    ]

    assert file_paths == ["README.md", "src/app.py"]
    assert chunk_ranges == [
        ("README.md", 1, 1),
        ("src/app.py", 1, 2),
        ("src/app.py", 3, 3),
    ]
    assert snapshot.repo_root == str(repo.resolve())
    assert snapshot.indexed_at > 0


def test_index_repository_saves_snapshot_to_json_store(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    source_file = repo / "main.py"
    source_file.write_bytes(b"def main():\n    return 'ok'\n")

    index_dir = tmp_path / ".code_context_index"

    snapshot = index_repository(
        repo,
        index_dir=index_dir,
        max_lines=10,
        overlap_lines=0,
    )

    store = JsonIndexStore(index_dir)
    loaded = store.load()

    assert store.exists() is True
    assert loaded == snapshot
    assert [file.relative_path for file in loaded.files] == ["main.py"]
    assert [chunk.relative_path for chunk in loaded.chunks] == ["main.py"]
    assert loaded.chunks[0].start_line == 1
    assert loaded.chunks[0].end_line == 2


def test_index_repository_supports_custom_index_filename(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    source_file = repo / "main.py"
    source_file.write_bytes(b"print('hello')\n")

    index_dir = tmp_path / ".code_context_index"

    index_repository(
        repo,
        index_dir=index_dir,
        max_lines=10,
        overlap_lines=0,
        index_filename="custom-index.json",
    )

    assert (index_dir / "custom-index.json").exists()


def test_build_repository_snapshot_supports_custom_extensions(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    python_file = repo / "main.py"
    python_file.write_bytes(b"print('ignored by custom extension list')\n")

    custom_file = repo / "notes.custom"
    custom_file.write_bytes(b"custom content\n")

    snapshot = build_repository_snapshot(
        repo,
        supported_extensions={".custom"},
        max_lines=10,
        overlap_lines=0,
    )

    assert [file.relative_path for file in snapshot.files] == ["notes.custom"]
    assert [chunk.relative_path for chunk in snapshot.chunks] == ["notes.custom"]
    assert snapshot.chunks[0].language == "unknown"


def test_chunk_scanned_files_uses_scanner_metadata(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    source_file = repo / "query.sql"
    source_file.write_bytes(b"select 1;\nselect 2;\nselect 3;\n")

    files = scan_repository(repo)
    chunks = chunk_scanned_files(files, max_lines=2, overlap_lines=0)

    assert [file.relative_path for file in files] == ["query.sql"]
    assert files[0].language == "sql"
    assert [(chunk.relative_path, chunk.language, chunk.start_line, chunk.end_line) for chunk in chunks] == [
        ("query.sql", "sql", 1, 2),
        ("query.sql", "sql", 3, 3),
    ]