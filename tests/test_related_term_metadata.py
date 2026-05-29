from __future__ import annotations

"""
ROLE: Verify matched related-term metadata across retrieval, ask, API, and CLI paths.
LAYER: tests
FLOW: related_term_metadata_validation

INPUTS:
- fuzzy developer questions
- explicit related terms
- temporary repository files
- indexed source chunks
- CLI and API ask requests

OUTPUTS:
- matched related-term metadata assertions
- API response metadata assertions
- CLI output metadata assertions

UPSTREAM:
- retrieval service
- ask workflow service
- FastAPI ask endpoint
- CLI ask command

DOWNSTREAM:
- local test runs
- future CI guardrails
- retrieval explainability behavior
- future demo checks

OWNS:
- matched related-term metadata tests
- related-term explainability assertions
- end-to-end API and CLI related-term metadata coverage

DOES_NOT_OWN:
- low-level vector scoring tests
- stale-index tests
- LLM generation tests
- clarification suggestion tests

SIDE_EFFECTS:
- writes temporary source files through pytest tmp_path
- writes temporary JSON index files through indexing pipeline
- captures CLI terminal output

STATE:
  reads:
    - temporary repository files
    - temporary JSON index files
  writes:
    - temporary repository files
    - temporary JSON index files
    - captured stdout and stderr

NOTES:
- Related terms are search aids, not source evidence.
- Matched related-term metadata only reports terms found in retrieved source chunks.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from code_context import cli
from code_context.api import create_app
from code_context.index_store import build_index_snapshot
from code_context.models import SourceChunk
from code_context.pipeline import index_repository
from code_context.retrieval import retrieve_grounded_context


def test_retrieval_reports_matched_related_terms() -> None:
    snapshot = build_index_snapshot(
        repo_path="C:/example/repo",
        files=[],
        chunks=[
            SourceChunk(
                chunk_id="src/code_context/api.py:1-5",
                relative_path="src/code_context/api.py",
                start_line=1,
                end_line=5,
                content=(
                    "def create_app():\n"
                    "    app = FastAPI()\n"
                    "    @app.post('/ask')\n"
                    "    def ask_question():\n"
                    "        return {'answer': 'ok'}\n"
                ),
                language="python",
            )
        ],
        indexed_at=123.45,
    )

    response = retrieve_grounded_context(
        snapshot=snapshot,
        query="What part lets another program talk to this?",
        related_terms=["HTTP", "API", "FastAPI", "endpoint", "route"],
        limit=1,
        min_score=0.0,
        minimum_results=1,
        minimum_top_score=0.0,
    )

    assert response.related_terms == ["HTTP", "API", "FastAPI", "endpoint", "route"]
    assert response.matched_related_terms == ["API", "FastAPI"]


def test_api_ask_reports_matched_related_terms(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    api_file = repo / "src" / "code_context" / "api.py"
    api_file.parent.mkdir(parents=True)
    api_file.write_bytes(
        b"def create_app():\n"
        b"    app = FastAPI()\n"
        b"    @app.post('/ask')\n"
        b"    def ask_question():\n"
        b"        return {'answer': 'ok'}\n"
    )

    index_dir = tmp_path / ".code_context_index"
    client = TestClient(create_app())

    index_response = client.post(
        "/index",
        json={
            "repo_path": str(repo),
            "index_dir": str(index_dir),
            "max_lines": 20,
            "overlap_lines": 0,
        },
    )
    assert index_response.status_code == 200

    ask_response = client.post(
        "/ask",
        json={
            "index_dir": str(index_dir),
            "question": "What part lets another program talk to this?",
            "related_terms": ["HTTP", "API", "FastAPI", "endpoint", "route"],
            "limit": 1,
            "min_score": 0.0,
            "minimum_results": 1,
            "minimum_top_score": 0.0,
        },
    )

    body = ask_response.json()

    assert ask_response.status_code == 200
    assert body["confidence"] == "grounded"
    assert body["related_terms"] == ["HTTP", "API", "FastAPI", "endpoint", "route"]
    assert body["matched_related_terms"] == ["API", "FastAPI"]


def test_cli_ask_prints_matched_related_terms(capsys, tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    api_file = repo / "src" / "code_context" / "api.py"
    api_file.parent.mkdir(parents=True)
    api_file.write_bytes(
        b"def create_app():\n"
        b"    app = FastAPI()\n"
        b"    @app.post('/ask')\n"
        b"    def ask_question():\n"
        b"        return {'answer': 'ok'}\n"
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
    assert "Related terms: HTTP, API, FastAPI, endpoint, route" in captured.out
    assert "Matched related terms: API, FastAPI" in captured.out
    assert captured.err == ""
