from __future__ import annotations

"""
ROLE: Verify reusable ask workflow orchestration behavior.
LAYER: tests
FLOW: ask_workflow_orchestration_validation

INPUTS:
- temporary repository directories
- temporary source files
- IndexSnapshot records
- developer question text
- retrieval thresholds
- modified source files that make indexed context stale
- monkeypatched optional workflow adapter calls

OUTPUTS:
- ask workflow service behavior assertions

UPSTREAM:
- ask workflow orchestration service
- optional LangGraph workflow adapter
- repository scanner
- drift detector
- IndexSnapshot model
- SourceChunk model

DOWNSTREAM:
- local test runs
- future CI guardrails
- FastAPI ask endpoint
- future CLI ask command
- future demo scripts
- system map review

OWNS:
- ask service grounded answer tests
- ask service insufficient-context tests
- ask service stale-index refusal tests
- ask service optional LangGraph adapter invocation tests

DOES_NOT_OWN:
- API route tests
- standalone scanner tests
- standalone chunker tests
- standalone drift tests
- standalone retrieval service tests
- standalone LangGraph adapter tests
- LLM response tests

SIDE_EFFECTS:
- writes temporary source files through pytest tmp_path
- modifies temporary source files to make indexed context stale
- monkeypatches the ask workflow adapter during one test

STATE:
  reads:
    - temporary repository files
    - in-memory test snapshots
  writes:
    - temporary repository files

NOTES:
- These tests keep ask orchestration reusable outside FastAPI.
- API tests should only verify HTTP response shaping after this service exists.
"""

from pathlib import Path

import pytest

import code_context.ask as ask_module
from code_context.agent.state import AgentStepStatus, VerificationResult, create_initial_state
from code_context.models import FileMetadata, IndexSnapshot, RetrievalResponse, SearchResult, SourceChunk
from code_context.scanner import calculate_file_hash


def test_ask_indexed_code_question_returns_grounded_result(tmp_path: Path) -> None:
    source_file = tmp_path / "scanner.py"
    source_file.write_bytes(
        b"def calculate_file_hash(path):\n"
        b"    return hashlib.sha256(path.read_bytes()).hexdigest()\n"
    )
    snapshot = _snapshot(
        repo_root=tmp_path,
        files=[_file_metadata(source_file, "scanner.py")],
        chunks=[
            _chunk(
                chunk_id="scanner.py:1-2",
                relative_path="scanner.py",
                content=(
                    "def calculate_file_hash(path):\n"
                    "    return hashlib.sha256(path.read_bytes()).hexdigest()\n"
                ),
            )
        ],
    )

    result = ask_module.ask_indexed_code_question(
        question="Where is calculate file hash handled?",
        snapshot=snapshot,
        limit=1,
        prefer_langgraph=False,
    )

    assert result.confidence == "grounded"
    assert result.is_grounded is True
    assert result.is_stale is False
    assert result.insufficient_reason is None
    assert result.answer
    assert result.plan
    assert result.sources[0].chunk.relative_path == "scanner.py"
    assert any("scanner.py" in citation for citation in result.citations)
    assert [step.status for step in result.steps] == [
        AgentStepStatus.completed,
        AgentStepStatus.completed,
        AgentStepStatus.completed,
        AgentStepStatus.completed,
    ]


def test_ask_indexed_code_question_returns_insufficient_context(tmp_path: Path) -> None:
    source_file = tmp_path / "api.py"
    source_file.write_bytes(b"def health():\n    return {'status': 'ok'}\n")
    snapshot = _snapshot(
        repo_root=tmp_path,
        files=[_file_metadata(source_file, "api.py")],
        chunks=[
            _chunk(
                chunk_id="api.py:1-2",
                relative_path="api.py",
                content="def health():\n    return {'status': 'ok'}\n",
            )
        ],
    )

    result = ask_module.ask_indexed_code_question(
        question="Where is the postgres liquibase migration handled?",
        snapshot=snapshot,
        limit=3,
        prefer_langgraph=False,
    )

    assert result.confidence == "insufficient_context"
    assert result.is_grounded is False
    assert result.is_stale is False
    assert result.insufficient_reason == "Not enough relevant indexed context was found."
    assert result.sources == []
    assert result.citations == []
    assert result.answer


def test_ask_indexed_code_question_refuses_when_snapshot_is_stale(tmp_path: Path) -> None:
    source_file = tmp_path / "scanner.py"
    source_file.write_bytes(
        b"def calculate_file_hash(path):\n"
        b"    return hashlib.sha256(path.read_bytes()).hexdigest()\n"
    )
    snapshot = _snapshot(
        repo_root=tmp_path,
        files=[_file_metadata(source_file, "scanner.py")],
        chunks=[
            _chunk(
                chunk_id="scanner.py:1-2",
                relative_path="scanner.py",
                content=(
                    "def calculate_file_hash(path):\n"
                    "    return hashlib.sha256(path.read_bytes()).hexdigest()\n"
                ),
            )
        ],
    )

    source_file.write_bytes(
        b"def calculate_file_hash(path):\n"
        b"    data = path.read_bytes()\n"
        b"    return hashlib.sha256(data).hexdigest()\n"
    )

    result = ask_module.ask_indexed_code_question(
        question="Where is calculate file hash handled?",
        snapshot=snapshot,
        limit=1,
        prefer_langgraph=False,
    )

    assert result.confidence == "stale_index"
    assert result.is_grounded is False
    assert result.is_stale is True
    assert result.insufficient_reason == "Indexed context is stale. Re-index the repository before asking questions."
    assert result.plan == []
    assert result.citations == []
    assert result.sources == []
    assert result.steps == []
    assert "Re-index the repository" in result.answer


def test_ask_indexed_code_question_passes_settings_to_optional_workflow_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source_file = tmp_path / "api.py"
    source_file.write_bytes(b"def health():\n    return {'status': 'ok'}\n")
    snapshot = _snapshot(
        repo_root=tmp_path,
        files=[_file_metadata(source_file, "api.py")],
        chunks=[
            _chunk(
                chunk_id="api.py:1-2",
                relative_path="api.py",
                content="def health():\n    return {'status': 'ok'}\n",
            )
        ],
    )
    calls: dict[str, object] = {}

    def fake_workflow_runner(
        *,
        question: str,
        snapshot: IndexSnapshot,
        limit: int,
        min_score: float,
        minimum_results: int,
        minimum_top_score: float,
        prefer_langgraph: bool,
    ):
        calls["question"] = question
        calls["snapshot"] = snapshot
        calls["limit"] = limit
        calls["min_score"] = min_score
        calls["minimum_results"] = minimum_results
        calls["minimum_top_score"] = minimum_top_score
        calls["prefer_langgraph"] = prefer_langgraph

        state = create_initial_state(question)
        completed_steps = [
            step.model_copy(update={"status": AgentStepStatus.completed})
            for step in state.steps
        ]
        retrieval = RetrievalResponse(
            query=question,
            is_sufficient=True,
            insufficient_reason=None,
            results=[
                SearchResult(
                    chunk=snapshot.chunks[0],
                    score=0.8,
                )
            ],
        )

        return state.model_copy(
            update={
                "plan": ["Use adapter."],
                "retrieval": retrieval,
                "verification": VerificationResult(can_answer=True, is_grounded=True),
                "answer": "fake answer",
                "citations": ["api.py:1-2"],
                "steps": completed_steps,
            }
        )

    monkeypatch.setattr(
        ask_module,
        "run_code_question_workflow_with_optional_langgraph",
        fake_workflow_runner,
    )

    result = ask_module.ask_indexed_code_question(
        question="Where is health handled?",
        snapshot=snapshot,
        limit=3,
        min_score=0.25,
        minimum_results=2,
        minimum_top_score=0.5,
        prefer_langgraph=True,
    )

    assert result.answer == "fake answer"
    assert result.confidence == "grounded"
    assert result.sources[0].chunk.relative_path == "api.py"

    assert calls["question"] == "Where is health handled?"
    assert calls["snapshot"] is snapshot
    assert calls["limit"] == 3
    assert calls["min_score"] == 0.25
    assert calls["minimum_results"] == 2
    assert calls["minimum_top_score"] == 0.5
    assert calls["prefer_langgraph"] is True


def _snapshot(
    *,
    repo_root: Path,
    files: list[FileMetadata],
    chunks: list[SourceChunk],
) -> IndexSnapshot:
    return IndexSnapshot(
        repo_root=str(repo_root.resolve()),
        indexed_at=456.0,
        files=files,
        chunks=chunks,
    )


def _file_metadata(path: Path, relative_path: str) -> FileMetadata:
    return FileMetadata(
        path=path,
        relative_path=relative_path,
        extension=path.suffix,
        size_bytes=path.stat().st_size,
        modified_at=path.stat().st_mtime,
        content_hash=calculate_file_hash(path),
        language="python",
    )


def _chunk(*, chunk_id: str, relative_path: str, content: str) -> SourceChunk:
    return SourceChunk(
        chunk_id=chunk_id,
        relative_path=relative_path,
        start_line=1,
        end_line=2,
        content=content,
        language="python",
    )