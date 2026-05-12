from __future__ import annotations

"""
ROLE: Verify optional LangGraph workflow adapter behavior.
LAYER: tests
FLOW: langgraph_agent_workflow_adapter_validation

INPUTS:
- developer question text
- sample IndexSnapshot records
- sample SourceChunk records
- retrieval thresholds
- optional LangGraph availability state

OUTPUTS:
- optional LangGraph adapter behavior assertions

UPSTREAM:
- optional LangGraph workflow adapter
- deterministic agent workflow runner
- agent state models
- IndexSnapshot model
- SourceChunk model

DOWNSTREAM:
- local test runs
- future CI guardrails
- future FastAPI ask endpoint LangGraph integration
- future demo scripts
- system map review

OWNS:
- optional LangGraph availability tests
- deterministic fallback tests
- LangGraph missing-dependency guardrail tests
- optional runner delegation tests

DOES_NOT_OWN:
- individual agent node tests
- retrieval service unit tests
- API route tests
- LLM answer tests
- ChromaDB tests

SIDE_EFFECTS:
- none

STATE:
  reads:
    - in-memory test snapshots
  writes:
    - none

NOTES:
- These tests keep LangGraph integration low-risk.
- The deterministic workflow should remain usable even when LangGraph is not installed.
- A later slice can switch API ask routing to this adapter after the LangGraph path is proven.
"""

from pathlib import Path

import pytest

from code_context.agent import langgraph_graph
from code_context.agent.graph import run_code_question_workflow
from code_context.agent.state import CodeQuestionState, create_initial_state
from code_context.models import FileMetadata, IndexSnapshot, SourceChunk


def test_optional_langgraph_runner_falls_back_to_deterministic_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(
        [
            _chunk(
                chunk_id="scanner.py:1-2",
                relative_path="scanner.py",
                content=(
                    "def calculate_file_hash(path):\n"
                    "    return hashlib.sha256(path.read_bytes()).hexdigest()\n"
                ),
            )
        ]
    )

    monkeypatch.setattr(langgraph_graph, "is_langgraph_available", lambda: False)

    result = langgraph_graph.run_code_question_workflow_with_optional_langgraph(
        question="Where is calculate file hash handled?",
        snapshot=snapshot,
        limit=1,
    )
    expected = run_code_question_workflow(
        question="Where is calculate file hash handled?",
        snapshot=snapshot,
        limit=1,
    )

    assert result.answer == expected.answer
    assert result.citations == expected.citations
    assert result.verification == expected.verification
    assert [step.status for step in result.steps] == [step.status for step in expected.steps]


def test_build_langgraph_workflow_raises_clear_error_when_dependency_is_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(langgraph_graph, "StateGraph", None)
    monkeypatch.setattr(langgraph_graph, "END", None)
    monkeypatch.setattr(
        langgraph_graph,
        "_LANGGRAPH_IMPORT_ERROR",
        ModuleNotFoundError("No module named 'langgraph'"),
    )

    with pytest.raises(RuntimeError, match="LangGraph is not installed"):
        langgraph_graph.build_langgraph_code_question_workflow()


def test_optional_langgraph_runner_delegates_to_langgraph_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _snapshot(
        [
            _chunk(
                chunk_id="api.py:1-2",
                relative_path="api.py",
                content="def health():\n    return {'status': 'ok'}\n",
            )
        ]
    )
    calls: dict[str, object] = {}

    def fake_langgraph_runner(
        *,
        question: str,
        snapshot: IndexSnapshot,
        limit: int,
        min_score: float,
        minimum_results: int,
        minimum_top_score: float,
    ) -> CodeQuestionState:
        calls["question"] = question
        calls["snapshot"] = snapshot
        calls["limit"] = limit
        calls["min_score"] = min_score
        calls["minimum_results"] = minimum_results
        calls["minimum_top_score"] = minimum_top_score
        return create_initial_state(question).model_copy(update={"answer": "fake langgraph answer"})

    monkeypatch.setattr(langgraph_graph, "is_langgraph_available", lambda: True)
    monkeypatch.setattr(
        langgraph_graph,
        "run_langgraph_code_question_workflow",
        fake_langgraph_runner,
    )

    result = langgraph_graph.run_code_question_workflow_with_optional_langgraph(
        question="Where is health handled?",
        snapshot=snapshot,
        limit=3,
        min_score=0.25,
        minimum_results=2,
        minimum_top_score=0.5,
    )

    assert result.answer == "fake langgraph answer"
    assert calls["question"] == "Where is health handled?"
    assert calls["snapshot"] is snapshot
    assert calls["limit"] == 3
    assert calls["min_score"] == 0.25
    assert calls["minimum_results"] == 2
    assert calls["minimum_top_score"] == 0.5


def _snapshot(chunks: list[SourceChunk]) -> IndexSnapshot:
    return IndexSnapshot(
        repo_root=str(Path("sample-repo")),
        indexed_at=456.0,
        files=[
            FileMetadata(
                path=Path("sample-repo/example.py"),
                relative_path="example.py",
                extension=".py",
                size_bytes=100,
                modified_at=123.0,
                content_hash="abc123",
                language="python",
            )
        ],
        chunks=chunks,
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