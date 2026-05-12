from __future__ import annotations

"""
ROLE: Verify end-to-end API ask workflows across indexing, grounded answering, stale refusal, and re-index recovery.
LAYER: tests
FLOW: api_ask_flow_validation

INPUTS:
- temporary repository directories
- temporary source files
- temporary index directories
- HTTP index requests
- HTTP ask requests
- modified source files that make indexed context stale

OUTPUTS:
- end-to-end ask workflow behavior assertions

UPSTREAM:
- FastAPI app implementation
- indexing pipeline
- JSON index store
- repository scanner
- drift detector
- deterministic agent workflow runner

DOWNSTREAM:
- local test runs
- future CI guardrails
- demo script validation
- future LangGraph workflow integration
- system map review

OWNS:
- end-to-end ask API flow tests
- index to ask success tests
- stale-index refusal flow tests
- re-index recovery tests
- API ask demo-path guardrails

DOES_NOT_OWN:
- standalone scanner tests
- standalone chunker tests
- standalone index store tests
- standalone vector store tests
- standalone retrieval service tests
- standalone agent workflow tests
- LLM response tests
- LangGraph integration tests

SIDE_EFFECTS:
- writes temporary source files through pytest tmp_path
- writes temporary JSON index files through API calls
- modifies temporary source files to make indexed context stale

STATE:
  reads:
    - temporary repository files
    - temporary JSON index files
  writes:
    - temporary repository files
    - temporary JSON index files

NOTES:
- These tests protect the local demo flow before LangGraph is added.
- The ask endpoint should answer only from current indexed context.
- Re-indexing should recover from stale-index refusal.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from code_context.api import create_app


def test_ask_flow_refuses_stale_index_then_recovers_after_reindex(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    source_file = repo / "scanner.py"
    source_file.write_bytes(
        b"def calculate_file_hash(path):\n"
        b"    return hashlib.sha256(path.read_bytes()).hexdigest()\n"
    )

    index_dir = tmp_path / ".code_context_index"
    client = TestClient(create_app())

    first_index_response = client.post(
        "/index",
        json={
            "repo_path": str(repo),
            "index_dir": str(index_dir),
            "max_lines": 10,
            "overlap_lines": 0,
        },
    )
    assert first_index_response.status_code == 200

    first_ask_response = client.post(
        "/ask",
        json={
            "index_dir": str(index_dir),
            "question": "Where is calculate file hash handled?",
            "limit": 1,
        },
    )
    first_ask_body = first_ask_response.json()

    assert first_ask_response.status_code == 200
    assert first_ask_body["confidence"] == "grounded"
    assert first_ask_body["is_grounded"] is True
    assert first_ask_body["is_stale"] is False
    assert first_ask_body["sources"][0]["relative_path"] == "scanner.py"

    source_file.write_bytes(
        b"def calculate_file_hash(path):\n"
        b"    data = path.read_bytes()\n"
        b"    return hashlib.sha256(data).hexdigest()\n"
    )

    stale_ask_response = client.post(
        "/ask",
        json={
            "index_dir": str(index_dir),
            "question": "Where is calculate file hash handled?",
            "limit": 1,
        },
    )
    stale_ask_body = stale_ask_response.json()

    assert stale_ask_response.status_code == 200
    assert stale_ask_body["confidence"] == "stale_index"
    assert stale_ask_body["is_grounded"] is False
    assert stale_ask_body["is_stale"] is True
    assert stale_ask_body["sources"] == []
    assert stale_ask_body["citations"] == []
    assert stale_ask_body["steps"] == []

    second_index_response = client.post(
        "/index",
        json={
            "repo_path": str(repo),
            "index_dir": str(index_dir),
            "max_lines": 10,
            "overlap_lines": 0,
        },
    )
    assert second_index_response.status_code == 200

    recovered_ask_response = client.post(
        "/ask",
        json={
            "index_dir": str(index_dir),
            "question": "Where is calculate file hash handled?",
            "limit": 1,
        },
    )
    recovered_ask_body = recovered_ask_response.json()

    assert recovered_ask_response.status_code == 200
    assert recovered_ask_body["confidence"] == "grounded"
    assert recovered_ask_body["is_grounded"] is True
    assert recovered_ask_body["is_stale"] is False
    assert recovered_ask_body["sources"][0]["relative_path"] == "scanner.py"
    assert [step["status"] for step in recovered_ask_body["steps"]] == [
        "completed",
        "completed",
        "completed",
        "completed",
    ]


def test_drift_and_ask_agree_when_index_is_stale(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    source_file = repo / "main.py"
    source_file.write_bytes(b"def main():\n    return 'before'\n")

    index_dir = tmp_path / ".code_context_index"
    client = TestClient(create_app())

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

    source_file.write_bytes(b"def main():\n    return 'after'\n")

    drift_response = client.post(
        "/drift",
        json={
            "repo_path": str(repo),
            "index_dir": str(index_dir),
        },
    )
    drift_body = drift_response.json()

    ask_response = client.post(
        "/ask",
        json={
            "index_dir": str(index_dir),
            "question": "Where is main handled?",
            "limit": 1,
        },
    )
    ask_body = ask_response.json()

    assert drift_response.status_code == 200
    assert drift_body["is_stale"] is True
    assert [item["relative_path"] for item in drift_body["modified"]] == ["main.py"]

    assert ask_response.status_code == 200
    assert ask_body["confidence"] == "stale_index"
    assert ask_body["is_stale"] is True
    assert ask_body["is_grounded"] is False
    assert ask_body["insufficient_reason"] == "Indexed context is stale. Re-index the repository before asking questions."