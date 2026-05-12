from __future__ import annotations

"""
ROLE: Verify FastAPI endpoints for health, indexing, search, retrieval, ask, and drift detection.
LAYER: tests
FLOW: api_validation

INPUTS:
- temporary repository directories
- temporary index directories
- HTTP request payloads
- indexed source snapshots
- deterministic ask questions

OUTPUTS:
- API response behavior assertions

UPSTREAM:
- FastAPI app implementation
- indexing pipeline
- JSON index store
- vector store
- retrieval service
- deterministic agent workflow runner
- drift detector

DOWNSTREAM:
- local test runs
- future CI guardrails
- demo scripts
- future agent workflow integration
- system map review

OWNS:
- API route tests
- request and response shape tests
- endpoint integration tests
- missing index HTTP behavior tests
- retrieve endpoint sufficiency tests
- ask endpoint grounded answer tests
- ask endpoint refusal tests
- drift endpoint tests

DOES_NOT_OWN:
- standalone scanner tests
- standalone chunker tests
- standalone index store tests
- standalone vector store tests
- standalone retrieval service tests
- standalone agent workflow tests
- LLM response tests

SIDE_EFFECTS:
- writes temporary source files through pytest tmp_path
- writes temporary JSON index files through API calls

STATE:
  reads:
    - temporary repository files
    - temporary JSON index files
  writes:
    - temporary repository files
    - temporary JSON index files

NOTES:
- These tests keep API endpoints thin and grounded in deterministic project behavior.
- The ask endpoint uses the deterministic agent workflow and does not generate LLM answers yet.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from code_context.api import create_app


def test_health_endpoint_returns_ok() -> None:
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_index_endpoint_indexes_repository_and_returns_summary(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    source_file = repo / "main.py"
    source_file.write_bytes(b"def main():\n    return 'ok'\n")

    index_dir = tmp_path / ".code_context_index"
    client = TestClient(create_app())

    response = client.post(
        "/index",
        json={
            "repo_path": str(repo),
            "index_dir": str(index_dir),
            "max_lines": 10,
            "overlap_lines": 0,
        },
    )

    body = response.json()

    assert response.status_code == 200
    assert body["repo_root"] == str(repo.resolve())
    assert body["file_count"] == 1
    assert body["chunk_count"] == 1
    assert body["indexed_at"] > 0
    assert Path(body["index_path"]).exists()


def test_search_endpoint_returns_grounded_chunk_results(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    scanner_file = repo / "scanner.py"
    scanner_file.write_bytes(
        b"def calculate_file_hash(path):\n"
        b"    return hashlib.sha256(path.read_bytes()).hexdigest()\n"
    )

    chunker_file = repo / "chunker.py"
    chunker_file.write_bytes(
        b"def chunk_text(content):\n"
        b"    return content.splitlines()\n"
    )

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

    search_response = client.post(
        "/search",
        json={
            "index_dir": str(index_dir),
            "query": "calculate file hash",
            "limit": 1,
        },
    )

    body = search_response.json()

    assert search_response.status_code == 200
    assert body["query"] == "calculate file hash"
    assert body["result_count"] == 1
    assert body["results"][0]["relative_path"] == "scanner.py"
    assert body["results"][0]["start_line"] == 1
    assert body["results"][0]["end_line"] == 2
    assert body["results"][0]["score"] > 0
    assert "calculate_file_hash" in body["results"][0]["content"]


def test_search_endpoint_returns_404_when_index_is_missing(tmp_path: Path) -> None:
    client = TestClient(create_app())

    response = client.post(
        "/search",
        json={
            "index_dir": str(tmp_path / "missing_index"),
            "query": "hash",
        },
    )

    assert response.status_code == 404
    assert "Index file does not exist" in response.json()["detail"]


def test_retrieve_endpoint_returns_sufficient_grounded_context(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    source_file = repo / "scanner.py"
    source_file.write_bytes(
        b"def calculate_file_hash(path):\n"
        b"    return hashlib.sha256(path.read_bytes()).hexdigest()\n"
    )

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

    retrieve_response = client.post(
        "/retrieve",
        json={
            "index_dir": str(index_dir),
            "query": "calculate file hash",
            "limit": 1,
        },
    )

    body = retrieve_response.json()

    assert retrieve_response.status_code == 200
    assert body["query"] == "calculate file hash"
    assert body["is_sufficient"] is True
    assert body["insufficient_reason"] is None
    assert body["result_count"] == 1
    assert body["results"][0]["relative_path"] == "scanner.py"


def test_retrieve_endpoint_reports_insufficient_context(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    source_file = repo / "api.py"
    source_file.write_bytes(b"def health():\n    return {'status': 'ok'}\n")

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

    retrieve_response = client.post(
        "/retrieve",
        json={
            "index_dir": str(index_dir),
            "query": "postgres liquibase migration",
            "limit": 3,
        },
    )

    body = retrieve_response.json()

    assert retrieve_response.status_code == 200
    assert body["is_sufficient"] is False
    assert body["insufficient_reason"] == "Not enough relevant indexed context was found."
    assert body["result_count"] == 0


def test_retrieve_endpoint_returns_404_when_index_is_missing(tmp_path: Path) -> None:
    client = TestClient(create_app())

    response = client.post(
        "/retrieve",
        json={
            "index_dir": str(tmp_path / "missing_index"),
            "query": "hash",
        },
    )

    assert response.status_code == 404
    assert "Index file does not exist" in response.json()["detail"]


def test_ask_endpoint_returns_grounded_answer_from_agent_workflow(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    scanner_file = repo / "scanner.py"
    scanner_file.write_bytes(
        b"def calculate_file_hash(path):\n"
        b"    return hashlib.sha256(path.read_bytes()).hexdigest()\n"
    )

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

    ask_response = client.post(
        "/ask",
        json={
            "index_dir": str(index_dir),
            "question": "Where is calculate file hash handled?",
            "limit": 1,
        },
    )

    body = ask_response.json()

    assert ask_response.status_code == 200
    assert body["question"] == "Where is calculate file hash handled?"
    assert body["confidence"] == "grounded"
    assert body["is_grounded"] is True
    assert body["insufficient_reason"] is None
    assert body["answer"]
    assert body["plan"]
    assert any("scanner.py" in citation for citation in body["citations"])
    assert body["sources"][0]["relative_path"] == "scanner.py"
    assert body["sources"][0]["start_line"] == 1
    assert body["sources"][0]["end_line"] == 2
    assert [step["status"] for step in body["steps"]] == [
        "completed",
        "completed",
        "completed",
        "completed",
    ]


def test_ask_endpoint_refuses_when_context_is_insufficient(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    source_file = repo / "api.py"
    source_file.write_bytes(b"def health():\n    return {'status': 'ok'}\n")

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

    ask_response = client.post(
        "/ask",
        json={
            "index_dir": str(index_dir),
            "question": "Where is the postgres liquibase migration handled?",
            "limit": 3,
        },
    )

    body = ask_response.json()

    assert ask_response.status_code == 200
    assert body["confidence"] == "insufficient_context"
    assert body["is_grounded"] is False
    assert body["insufficient_reason"] == "Not enough relevant indexed context was found."
    assert body["sources"] == []
    assert body["citations"] == []
    assert body["answer"]


def test_ask_endpoint_returns_404_when_index_is_missing(tmp_path: Path) -> None:
    client = TestClient(create_app())

    response = client.post(
        "/ask",
        json={
            "index_dir": str(tmp_path / "missing_index"),
            "question": "Where is hash calculation handled?",
        },
    )

    assert response.status_code == 404
    assert "Index file does not exist" in response.json()["detail"]


def test_drift_endpoint_reports_modified_file_after_indexing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    source_file = repo / "main.py"
    source_file.write_bytes(b"print('before')\n")

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

    source_file.write_bytes(b"print('after')\n")

    drift_response = client.post(
        "/drift",
        json={
            "repo_path": str(repo),
            "index_dir": str(index_dir),
        },
    )

    body = drift_response.json()

    assert drift_response.status_code == 200
    assert body["is_stale"] is True
    assert [item["relative_path"] for item in body["modified"]] == ["main.py"]
    assert body["added"] == []
    assert body["removed"] == []


def test_drift_endpoint_returns_404_when_index_is_missing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    client = TestClient(create_app())

    response = client.post(
        "/drift",
        json={
            "repo_path": str(repo),
            "index_dir": str(tmp_path / "missing_index"),
        },
    )

    assert response.status_code == 404
    assert "Index file does not exist" in response.json()["detail"]