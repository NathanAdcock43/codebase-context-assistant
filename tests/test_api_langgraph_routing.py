from __future__ import annotations

"""
ROLE: Verify that the ask API route uses the optional LangGraph workflow adapter.
LAYER: tests
FLOW: api_langgraph_routing_validation

INPUTS:
- temporary repository directories
- temporary source files
- temporary index directories
- HTTP index requests
- HTTP ask requests
- monkeypatched optional LangGraph workflow adapter

OUTPUTS:
- API routing behavior assertions for ask workflow execution

UPSTREAM:
- FastAPI app implementation
- optional LangGraph workflow adapter
- agent state models
- indexing pipeline
- JSON index store

DOWNSTREAM:
- local test runs
- future CI guardrails
- future LangGraph ask endpoint integration
- future demo scripts
- system map review

OWNS:
- ask endpoint workflow routing tests
- optional LangGraph adapter invocation tests
- API to agent adapter boundary tests

DOES_NOT_OWN:
- standalone LangGraph adapter tests
- deterministic agent workflow tests
- retrieval service unit tests
- drift endpoint tests
- LLM response tests

SIDE_EFFECTS:
- writes temporary source files through pytest tmp_path
- writes temporary JSON index files through API calls
- monkeypatches the API workflow adapter during the test

STATE:
  reads:
    - temporary repository files
    - temporary JSON index files
  writes:
    - temporary repository files
    - temporary JSON index files

NOTES:
- This test protects the seam between the API and the optional LangGraph adapter.
- The adapter itself still owns fallback behavior when LangGraph is unavailable.
"""

from pathlib import Path

from fastapi.testclient import TestClient

import code_context.api as api_module
from code_context.agent.state import AgentStepStatus, CodeQuestionState, VerificationResult, create_initial_state
from code_context.models import IndexSnapshot


def test_ask_endpoint_routes_through_optional_langgraph_adapter(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    source_file = repo / "api.py"
    source_file.write_bytes(b"def health():\n    return {'status': 'ok'}\n")

    index_dir = tmp_path / ".code_context_index"
    client = TestClient(api_module.create_app())
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
    ) -> CodeQuestionState:
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

        return state.model_copy(
            update={
                "plan": ["Use the optional LangGraph workflow adapter."],
                "verification": VerificationResult(can_answer=True, is_grounded=True),
                "answer": "fake optional langgraph answer",
                "citations": ["api.py:1-2"],
                "steps": completed_steps,
            }
        )

    monkeypatch.setattr(
        api_module,
        "run_code_question_workflow_with_optional_langgraph",
        fake_workflow_runner,
    )

    index_response = client.post(
        "/index",
        json={
            "repo_path": str(repo),
            "index_dir": str(index_dir),
            "max_lines": 10,
            "overlap_lines": 0,
        },
    )
    assert index_response.status_code == 200

    ask_response = client.post(
        "/ask",
        json={
            "index_dir": str(index_dir),
            "question": "Where is health handled?",
            "limit": 3,
            "min_score": 0.25,
            "minimum_results": 2,
            "minimum_top_score": 0.5,
        },
    )

    body = ask_response.json()

    assert ask_response.status_code == 200
    assert body["answer"] == "fake optional langgraph answer"
    assert body["confidence"] == "grounded"
    assert body["is_grounded"] is True
    assert body["is_stale"] is False
    assert body["plan"] == ["Use the optional LangGraph workflow adapter."]
    assert body["citations"] == ["api.py:1-2"]

    assert calls["question"] == "Where is health handled?"
    assert isinstance(calls["snapshot"], IndexSnapshot)
    assert calls["limit"] == 3
    assert calls["min_score"] == 0.25
    assert calls["minimum_results"] == 2
    assert calls["minimum_top_score"] == 0.5
    assert calls["prefer_langgraph"] is True