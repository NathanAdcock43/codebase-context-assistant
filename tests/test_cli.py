from __future__ import annotations

"""
ROLE: Verify local CLI index, ask, generated ask, and drift command behavior.
LAYER: tests
FLOW: cli_validation

INPUTS:
- temporary repository directories
- temporary source files
- temporary index directories
- CLI argument lists
- existing JSON index snapshots
- modified source files that make indexed context stale
- monkeypatched ask workflow service calls for generated-answer routing

OUTPUTS:
- CLI exit code assertions
- CLI stdout assertions
- CLI stderr assertions
- persisted index assertions
- generated-answer CLI routing assertions
- terminal drift report assertions
- ticket-style CLI ask grounding assertions

UPSTREAM:
- CLI implementation
- repository indexing pipeline
- reusable ask workflow service
- drift detector
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
- ticket-style CLI ask tests
- CLI generated-answer option tests
- CLI drift command tests
- terminal output formatting tests
- missing repository error tests
- missing index error tests
- stale-index refusal output tests
- drift report output tests

DOES_NOT_OWN:
- ask service unit tests
- API route tests
- repository scanner tests
- retrieval service tests
- standalone drift detector tests
- LangGraph adapter tests
- real LLM provider tests

SIDE_EFFECTS:
- writes temporary source files through pytest tmp_path
- writes temporary JSON index files through CLI and indexing pipeline
- modifies temporary source files to make indexed context stale
- monkeypatches ask service in generated-answer routing tests
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
- Generated-answer CLI tests do not call a real LLM provider.
- The index and drift commands give the project a simple terminal-first demo path.
"""

from pathlib import Path
from typing import Any

from code_context import cli
from code_context.agent.state import AgentStepStatus, create_initial_state
from code_context.ask import AskWorkflowResult
from code_context.index_store import JsonIndexStore
from code_context.models import IndexSnapshot
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
    assert "LLM generated: no" in captured.out
    assert "scanner.py" in captured.out
    assert "Workflow steps:" in captured.out
    assert captured.err == ""



def test_cli_ask_uses_ticket_anchor_retrieval_and_prunes_weak_sources(
    capsys,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    export_file = repo / "src" / "exports" / "options.py"
    menu_file = repo / "src" / "navigation" / "menu.py"
    export_file.parent.mkdir(parents=True)
    menu_file.parent.mkdir(parents=True)

    export_file.write_bytes(
        b"class ExportRunOptions:\n"
        b"    CUSTOMER_EXPORT_STATUS = 'N'\n"
        b"    def apply_export_status(self):\n"
        b"        return CUSTOMER_EXPORT_STATUS\n"
    )
    menu_file.write_bytes(
        b"class NavigationMenu:\n"
        b"    def render_user_menu(self):\n"
        b"        return 'menu'\n"
    )

    index_dir = tmp_path / ".code_context_index"
    index_repository(
        repo,
        index_dir=index_dir,
        max_lines=20,
        overlap_lines=0,
    )

    exit_code = cli.main(
        [
            "ask",
            "--index-dir",
            str(index_dir),
            "--question",
            (
                "TASK-123\n"
                "Data change request\n"
                "Add CUSTOMER_EXPORT_STATUS to EXPORT_RUN_OPTIONS.\n"
                "Users need an option to control export status during run setup."
            ),
            "--limit",
            "2",
            "--min-score",
            "0",
            "--minimum-results",
            "1",
            "--minimum-top-score",
            "0",
            "--no-langgraph",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Confidence: grounded" in captured.out
    assert "Grounded: yes" in captured.out
    assert "Stale: no" in captured.out
    assert "LLM generated: no" in captured.out
    assert "src/exports/options.py" in captured.out
    assert "src/navigation/menu.py" not in captured.out
    assert captured.err == ""


def test_cli_ask_accepts_related_terms_for_fuzzy_question(
    capsys,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    api_file = repo / "src" / "code_context" / "api.py"
    docs_file = repo / "docs" / "project_brief.md"
    api_file.parent.mkdir(parents=True)
    docs_file.parent.mkdir(parents=True)

    api_file.write_bytes(
        b"def create_app():\n"
        b"    app = FastAPI()\n"
        b"    @app.post('/ask')\n"
        b"    def ask_question():\n"
        b"        return {'answer': 'ok'}\n"
    )
    docs_file.write_bytes(
        b"This project helps developers understand code.\n"
        b"It scans files and answers questions.\n"
    )

    index_dir = tmp_path / ".code_context_index"
    index_repository(
        repo,
        index_dir=index_dir,
        max_lines=20,
        overlap_lines=0,
    )

    exit_code = cli.main(
        [
            "ask",
            "--index-dir",
            str(index_dir),
            "--question",
            "What part lets another program talk to this?",
            "--related-term",
            "HTTP",
            "--related-term",
            "API",
            "--related-terms",
            "FastAPI,endpoint,route",
            "--limit",
            "2",
            "--min-score",
            "0",
            "--minimum-results",
            "1",
            "--minimum-top-score",
            "0",
            "--no-langgraph",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Question: What part lets another program talk to this?" in captured.out
    assert "Related terms: HTTP, API, FastAPI, endpoint, route" in captured.out
    assert "Confidence: grounded" in captured.out
    assert "src/code_context/api.py" in captured.out
    assert captured.err == ""




def test_cli_ask_prints_clarification_for_grounded_docs_only_fuzzy_question(
    capsys,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    docs_file = repo / "docs" / "project_brief.md"
    docs_file.parent.mkdir(parents=True)
    docs_file.write_bytes(
        b"The project exposes a FastAPI API for codebase questions.\n"
    )

    index_dir = tmp_path / ".code_context_index"
    index_repository(
        repo,
        index_dir=index_dir,
        max_lines=20,
        overlap_lines=0,
    )

    exit_code = cli.main(
        [
            "ask",
            "--index-dir",
            str(index_dir),
            "--question",
            "What part lets another program talk to this?",
            "--limit",
            "1",
            "--min-score",
            "0",
            "--minimum-results",
            "1",
            "--minimum-top-score",
            "0",
            "--no-langgraph",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Confidence: needs_clarification" in captured.out
    assert "Clarification suggestion:" in captured.out
    assert "Suggested related terms: HTTP, API, FastAPI, endpoint, route" in captured.out
    assert "Retrieved context did not include implementation source files." in captured.out
    assert captured.err == ""


def test_cli_ask_prints_clarification_suggestion_for_fuzzy_question(
    capsys,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    docs_file = repo / "docs" / "project_brief.md"
    docs_file.parent.mkdir(parents=True)
    docs_file.write_bytes(
        b"This project scans files and answers codebase questions.\n"
    )

    index_dir = tmp_path / ".code_context_index"
    index_repository(
        repo,
        index_dir=index_dir,
        max_lines=20,
        overlap_lines=0,
    )

    exit_code = cli.main(
        [
            "ask",
            "--index-dir",
            str(index_dir),
            "--question",
            "What part lets another program talk to this?",
            "--limit",
            "1",
            "--min-score",
            "0.99",
            "--minimum-results",
            "1",
            "--minimum-top-score",
            "0",
            "--no-langgraph",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Confidence: needs_clarification" in captured.out
    assert "Clarification suggestion:" in captured.out
    assert "Suggested related terms: HTTP, API, FastAPI, endpoint, route" in captured.out
    assert "Where is the HTTP API layer implemented" in captured.out
    assert captured.err == ""


def test_cli_ask_routes_generated_answer_options_to_ask_service(
    monkeypatch: Any,
    capsys: Any,
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
    calls: dict[str, object] = {}

    def fake_ask_service(
        *,
        question: str,
        snapshot: IndexSnapshot,
        limit: int,
        min_score: float,
        minimum_results: int,
        minimum_top_score: float,
        prefer_langgraph: bool,
        related_terms: list[str] | None,
        use_llm: bool,
        llm_model: str | None,
        llm_temperature: float | None,
    ) -> AskWorkflowResult:
        calls["question"] = question
        calls["snapshot"] = snapshot
        calls["limit"] = limit
        calls["min_score"] = min_score
        calls["minimum_results"] = minimum_results
        calls["minimum_top_score"] = minimum_top_score
        calls["prefer_langgraph"] = prefer_langgraph
        calls["related_terms"] = related_terms
        calls["use_llm"] = use_llm
        calls["llm_model"] = llm_model
        calls["llm_temperature"] = llm_temperature

        state = create_initial_state(question)
        completed_steps = [
            step.model_copy(update={"status": AgentStepStatus.completed})
            for step in state.steps
        ]

        return AskWorkflowResult(
            question=question,
            answer="fake generated CLI answer",
            confidence="grounded_generated",
            is_grounded=True,
            is_stale=False,
            insufficient_reason=None,
            plan=["Use generated answer path."],
            citations=["scanner.py:1-2"],
            sources=[],
            steps=completed_steps,
            is_llm_generated=True,
            llm_provider="fake",
            llm_model="fake-model",
            llm_usage={"input_tokens": 12, "output_tokens": 8},
        )

    monkeypatch.setattr(cli, "ask_indexed_code_question", fake_ask_service)

    exit_code = cli.main(
        [
            "ask",
            "--index-dir",
            str(index_dir),
            "--question",
            "Where is calculate file hash handled?",
            "--limit",
            "3",
            "--min-score",
            "0.25",
            "--minimum-results",
            "2",
            "--minimum-top-score",
            "0.5",
            "--no-langgraph",
            "--use-llm",
            "--llm-model",
            "fake-model",
            "--llm-temperature",
            "0.1",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Confidence: grounded_generated" in captured.out
    assert "LLM generated: yes" in captured.out
    assert "LLM provider: fake" in captured.out
    assert "LLM model: fake-model" in captured.out
    assert "LLM usage: {'input_tokens': 12, 'output_tokens': 8}" in captured.out
    assert "fake generated CLI answer" in captured.out
    assert captured.err == ""

    assert calls["question"] == "Where is calculate file hash handled?"
    assert isinstance(calls["snapshot"], IndexSnapshot)
    assert calls["limit"] == 3
    assert calls["min_score"] == 0.25
    assert calls["minimum_results"] == 2
    assert calls["minimum_top_score"] == 0.5
    assert calls["prefer_langgraph"] is False
    assert calls["related_terms"] == []
    assert calls["use_llm"] is True
    assert calls["llm_model"] == "fake-model"
    assert calls["llm_temperature"] == 0.1


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
    assert "LLM generated: no" in captured.out
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


def test_cli_drift_reports_current_index(
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
            "drift",
            "--index-dir",
            str(index_dir),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Drift status: current" in captured.out
    assert "Stale: no" in captured.out
    assert "Added files: 0" in captured.out
    assert "Modified files: 0" in captured.out
    assert "Removed files: 0" in captured.out
    assert "scanner.py" not in captured.out
    assert captured.err == ""


def test_cli_drift_reports_added_modified_and_removed_files(
    capsys,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    modified_file = repo / "scanner.py"
    removed_file = repo / "old.py"

    modified_file.write_bytes(
        b"def calculate_file_hash(path):\n"
        b"    return hashlib.sha256(path.read_bytes()).hexdigest()\n"
    )
    removed_file.write_bytes(b"def old_function():\n    return None\n")

    index_dir = tmp_path / ".code_context_index"
    index_repository(
        repo,
        index_dir=index_dir,
        max_lines=10,
        overlap_lines=0,
    )

    modified_file.write_bytes(
        b"def calculate_file_hash(path):\n"
        b"    data = path.read_bytes()\n"
        b"    return hashlib.sha256(data).hexdigest()\n"
    )
    removed_file.unlink()
    added_file = repo / "new.py"
    added_file.write_bytes(b"def new_function():\n    return True\n")

    exit_code = cli.main(
        [
            "drift",
            "--index-dir",
            str(index_dir),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Drift status: stale" in captured.out
    assert "Stale: yes" in captured.out
    assert "Added files: 1" in captured.out
    assert "Modified files: 1" in captured.out
    assert "Removed files: 1" in captured.out
    assert "Added:" in captured.out
    assert "- new.py" in captured.out
    assert "Modified:" in captured.out
    assert "- scanner.py" in captured.out
    assert "Removed:" in captured.out
    assert "- old.py" in captured.out
    assert captured.err == ""


def test_cli_drift_can_show_unchanged_files(
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
            "drift",
            "--index-dir",
            str(index_dir),
            "--show-unchanged",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Drift status: current" in captured.out
    assert "Unchanged files: 1" in captured.out
    assert "Unchanged:" in captured.out
    assert "- scanner.py" in captured.out
    assert captured.err == ""


def test_cli_drift_returns_error_for_missing_index(
    capsys,
    tmp_path: Path,
) -> None:
    missing_index_dir = tmp_path / ".missing_index"

    exit_code = cli.main(
        [
            "drift",
            "--index-dir",
            str(missing_index_dir),
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 1
    assert captured.out == ""
    assert "Error:" in captured.err