from __future__ import annotations

"""
ROLE: Verify that the ask API route uses the reusable ask workflow service.
LAYER: tests
FLOW: api_ask_service_routing_validation

INPUTS:
- temporary repository directories
- temporary source files
- temporary index directories
- HTTP index requests
- HTTP ask requests
- monkeypatched ask workflow service
- optional generated-answer request fields

OUTPUTS:
- API routing behavior assertions for ask workflow execution
- API generated-answer routing assertions
- ask response LLM metadata shaping assertions

UPSTREAM:
- FastAPI app implementation
- ask workflow orchestration service
- agent state models
- indexing pipeline
- JSON index store

DOWNSTREAM:
- local test runs
- future CI guardrails
- future CLI ask command
- future demo scripts
- system map review

OWNS:
- ask endpoint workflow service routing tests
- API to ask service boundary tests
- ask response shaping tests after service extraction
- API opt-in generated answer routing tests

DOES_NOT_OWN:
- standalone ask service tests
- standalone LangGraph adapter tests
- deterministic agent workflow tests
- retrieval service unit tests
- drift endpoint tests
- real LLM provider tests

SIDE_EFFECTS:
- writes temporary source files through pytest tmp_path
- writes temporary JSON index files through API calls
- monkeypatches the API ask workflow service during tests

STATE:
  reads:
    - temporary repository files
    - temporary JSON index files
  writes:
    - temporary repository files
    - temporary JSON index files

NOTES:
- This test protects the seam between the API and the reusable ask service.
- The ask service owns optional LangGraph routing, deterministic fallback behavior, and generated answer guardrails.
- These tests do not call a real LLM provider.
"""

from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

import code_context.api as api_module
from code_context.agent.state import AgentStepStatus, create_initial_state
from code_context.ask import AskWorkflowResult
from code_context.models import IndexSnapshot


def test_ask_endpoint_routes_through_ask_workflow_service(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    source_file = repo / "api.py"
    source_file.write_bytes(b"def health():\n    return {'status': 'ok'}\n")

    index_dir = tmp_path / ".code_context_index"
    client = TestClient(api_module.create_app())
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
            answer="fake ask service answer",
            confidence="grounded",
            is_grounded=True,
            is_stale=False,
            insufficient_reason=None,
            plan=["Use the reusable ask workflow service."],
            citations=["api.py:1-2"],
            sources=[],
            steps=completed_steps,
        )

    monkeypatch.setattr(
        api_module,
        "ask_indexed_code_question",
        fake_ask_service,
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
    assert body["answer"] == "fake ask service answer"
    assert body["confidence"] == "grounded"
    assert body["is_grounded"] is True
    assert body["is_stale"] is False
    assert body["is_llm_generated"] is False
    assert body["llm_provider"] is None
    assert body["llm_model"] is None
    assert body["llm_usage"] == {}
    assert body["plan"] == ["Use the reusable ask workflow service."]
    assert body["citations"] == ["api.py:1-2"]
    assert body["sources"] == []
    assert [step["status"] for step in body["steps"]] == [
        "completed",
        "completed",
        "completed",
        "completed",
    ]

    assert calls["question"] == "Where is health handled?"
    assert isinstance(calls["snapshot"], IndexSnapshot)
    assert calls["limit"] == 3
    assert calls["min_score"] == 0.25
    assert calls["minimum_results"] == 2
    assert calls["minimum_top_score"] == 0.5
    assert calls["prefer_langgraph"] is True
    assert calls["use_llm"] is False
    assert calls["llm_model"] is None
    assert calls["llm_temperature"] is None


def test_ask_endpoint_routes_generated_answer_request_fields_to_ask_service(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    source_file = repo / "api.py"
    source_file.write_bytes(b"def health():\n    return {'status': 'ok'}\n")

    index_dir = tmp_path / ".code_context_index"
    client = TestClient(api_module.create_app())
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
        use_llm: bool,
        llm_model: str | None,
        llm_temperature: float,
    ) -> AskWorkflowResult:
        calls["question"] = question
        calls["snapshot"] = snapshot
        calls["limit"] = limit
        calls["min_score"] = min_score
        calls["minimum_results"] = minimum_results
        calls["minimum_top_score"] = minimum_top_score
        calls["prefer_langgraph"] = prefer_langgraph
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
            answer="fake generated ask service answer",
            confidence="grounded_generated",
            is_grounded=True,
            is_stale=False,
            insufficient_reason=None,
            plan=["Use the reusable ask workflow service."],
            citations=["api.py:1-2"],
            sources=[],
            steps=completed_steps,
            is_llm_generated=True,
            llm_provider="fake",
            llm_model="fake-model",
            llm_usage={"input_tokens": 12, "output_tokens": 8},
        )

    monkeypatch.setattr(
        api_module,
        "ask_indexed_code_question",
        fake_ask_service,
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
            "use_llm": True,
            "llm_model": "fake-model",
            "llm_temperature": 0.1,
        },
    )

    body = ask_response.json()

    assert ask_response.status_code == 200
    assert body["answer"] == "fake generated ask service answer"
    assert body["confidence"] == "grounded_generated"
    assert body["is_grounded"] is True
    assert body["is_stale"] is False
    assert body["is_llm_generated"] is True
    assert body["llm_provider"] == "fake"
    assert body["llm_model"] == "fake-model"
    assert body["llm_usage"] == {"input_tokens": 12, "output_tokens": 8}

    assert calls["question"] == "Where is health handled?"
    assert isinstance(calls["snapshot"], IndexSnapshot)
    assert calls["limit"] == 3
    assert calls["min_score"] == 0.25
    assert calls["minimum_results"] == 2
    assert calls["minimum_top_score"] == 0.5
    assert calls["prefer_langgraph"] is True
    assert calls["use_llm"] is True
    assert calls["llm_model"] == "fake-model"
    assert calls["llm_temperature"] == 0.1
