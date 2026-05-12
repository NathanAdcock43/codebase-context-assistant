from __future__ import annotations

"""
ROLE: Verify local CLI index and ask command behavior.
LAYER: tests
FLOW: cli_validation

INPUTS:
- temporary repository directories
- temporary source files
- temporary index directories
- CLI argument lists
- existing JSON index snapshots

OUTPUTS:
- CLI exit code assertions
- CLI stdout assertions
- CLI stderr assertions
- persisted index assertions

UPSTREAM:
- CLI implementation
- repository indexing pipeline
- reusable ask workflow service
- JSON index store

DOWNSTREAM:
- local test runs
- future CI guardrails
- README command examples
- demo script validation
- system map review

OWNS:
- CLI index command tests
- CLI ask command tests
- terminal output formatting tests
- missing repository error tests
- missing index error tests
- stale-index refusal output tests

DOES_NOT_OWN:
- ask service unit tests
- API route tests
- repository scanner tests
- retrieval service tests
- LangGraph adapter tests
- LLM response tests

SIDE_EFFECTS:
- writes temporary source files through pytest tmp_path
- writes temporary JSON index files through CLI and indexing pipeline
- modifies temporary source files to make indexed context stale
- captures stdout and stderr through pytest

STATE:
  reads:
    - temporary repository files
    - temporary JSON index files
  writes:
    - temporary repository files
    - temporary JSON index files
    - captured stdout
    - captured stderr

NOTES:
- These tests keep the CLI thin and focused on service wiring.
- Stale-index refusal is a successful CLI execution because the tool refused correctly.
- The index command gives the project a simple terminal-first demo path.
"""

from pathlib import Path

from code_context import cli
from code_context.index_store import JsonIndexStore
from code_context.pipeline import index_repository


def test_cli_index_creates_index_and_prints_summary(
    capsys,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    source_file = repo / "scanner.py"
    source_file.write_bytes(
        b"def calculate_file_hash(path):\n"
        b"    return hashlib.sha256(path.read_bytes()).hexdigest()\n"
    )

    index_dir = tmp_path / ".code_context_index"

    exit_code = cli.main(
        [
            "index",
            "--repo",
            str(repo),
            "--index-dir",
            str(index_dir),
            "--max-lines",
            "10",
            "--overlap-lines",
            "0",
        ]
    )

    captured = capsys.readouterr()
    snapshot = JsonIndexStore(index_dir).load()

    assert exit_code == 0
    assert "Indexed repository:" in captured.out
    assert str(repo.resolve()) in captured.out
    assert "Index path:" in captured.out
    assert "File count: 1" in captured.out
    assert "Chunk count: 1" in captured.out
    assert captured.err == ""

    assert snapshot.repo_root == str(repo.resolve())
    assert len(snapshot.files) == 1
    assert len(snapshot.chunks) == 1
    assert snapshot.chunks[0].relative_path == "scanner.py"


def test_cli_index_supports_custom_index_filename(
    capsys,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    source_file = repo / "api.py"
    source_file.write_bytes(b"def health():\n    return {'status': 'ok'}\n")

    index_dir = tmp_path / ".code_context_index"

    exit_code = cli.main(
        [
            "index",
            "--repo",
            str(repo),
            "--index-dir",
            str(index_dir),
            "--index-filename",
            "custom-index.json",
        ]
    )

    captured = capsys.readouterr()
    snapshot = JsonIndexStore(index_dir, index_filename="custom-index.json").load()

    assert exit_code == 0
    assert "custom-index.json" in captured.out
    assert captured.err == ""
    assert len(snapshot.files) == 1
    assert len(snapshot.chunks) == 1


def test_cli_index_returns_error_for_missing_repository(
    capsys,
    tmp_path: Path,
) -> None:
    missing_repo = tmp_path / "missing-repo"
    index_dir = tmp_path / ".code_context_index"

    exit_code = cli.main(
        [
            "index",
            "--repo",
            str(missing_repo),
            "--index-dir",
            str(index_dir),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "Error:" in captured.err


def test_cli_ask_prints_grounded_answer_for_existing_index(
    capsys,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    source_file = repo / "scanner.py"
    source_file.write_bytes(
        b"def calculate_file_hash(path):\n"
        b"    return hashlib.sha256(path.read_bytes()).hexdigest()\n"
    )

    index_dir = tmp_path / ".code_context_index"
    index_repository(
        repo,
        index_dir=index_dir,
        max_lines=10,
        overlap_lines=0,
    )

    exit_code = cli.main(
        [
            "ask",
            "--index-dir",
            str(index_dir),
            "--question",
            "Where is calculate file hash handled?",
            "--limit",
            "1",
            "--no-langgraph",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Question: Where is calculate file hash handled?" in captured.out
    assert "Confidence: grounded" in captured.out
    assert "Grounded: yes" in captured.out
    assert "Stale: no" in captured.out
    assert "scanner.py" in captured.out
    assert "Workflow steps:" in captured.out
    assert captured.err == ""


def test_cli_ask_prints_stale_refusal_without_error_exit(
    capsys,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    source_file = repo / "scanner.py"
    source_file.write_bytes(
        b"def calculate_file_hash(path):\n"
        b"    return hashlib.sha256(path.read_bytes()).hexdigest()\n"
    )

    index_dir = tmp_path / ".code_context_index"
    index_repository(
        repo,
        index_dir=index_dir,
        max_lines=10,
        overlap_lines=0,
    )

    source_file.write_bytes(
        b"def calculate_file_hash(path):\n"
        b"    data = path.read_bytes()\n"
        b"    return hashlib.sha256(data).hexdigest()\n"
    )

    exit_code = cli.main(
        [
            "ask",
            "--index-dir",
            str(index_dir),
            "--question",
            "Where is calculate file hash handled?",
            "--limit",
            "1",
            "--no-langgraph",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Confidence: stale_index" in captured.out
    assert "Grounded: no" in captured.out
    assert "Stale: yes" in captured.out
    assert "Re-index the repository" in captured.out
    assert captured.err == ""


def test_cli_ask_returns_error_for_missing_index(
    capsys,
    tmp_path: Path,
) -> None:
    missing_index_dir = tmp_path / ".missing_index"

    exit_code = cli.main(
        [
            "ask",
            "--index-dir",
            str(missing_index_dir),
            "--question",
            "Where is scanner handled?",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "Error:" in captured.err