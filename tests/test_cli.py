from __future__ import annotations

"""
ROLE: Verify local CLI ask command behavior.
LAYER: tests
FLOW: cli_ask_validation

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
- CLI ask command tests
- terminal output formatting tests
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
- writes temporary JSON index files through indexing pipeline
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
"""

from pathlib import Path

from code_context import cli
from code_context.pipeline import index_repository


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