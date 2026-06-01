from __future__ import annotations

"""
ROLE: Verify enrichment usage metadata across retrieval, API, and CLI paths.
LAYER: tests
FLOW: enrichment_metadata_validation

INPUTS:
- temporary repository files
- file enrichment metadata
- fuzzy developer questions
- API ask requests
- CLI ask commands

OUTPUTS:
- retrieval enrichment usage assertions
- API enrichment metadata assertions
- CLI enrichment metadata assertions

UPSTREAM:
- retrieval service
- index store
- FastAPI ask endpoint
- CLI ask command

DOWNSTREAM:
- local test runs
- future CI guardrails
- enrichment explainability behavior

OWNS:
- enrichment usage metadata tests
- matched enrichment term tests
- end-to-end API and CLI enrichment metadata coverage

DOES_NOT_OWN:
- provider-backed enrichment generation
- LLM calls
- ChromaDB behavior
- stale-index tests

SIDE_EFFECTS:
- writes temporary repository files through pytest tmp_path
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
- Enrichment metadata is a search aid, not source evidence.
- Answers should still cite returned source chunks, not enrichment records.
"""

from pathlib import Path

from fastapi.testclient import TestClient

from code_context import cli
from code_context.api import create_app
from code_context.index_store import JsonIndexStore
from code_context.models import FileEnrichment
from code_context.pipeline import index_repository
from code_context.retrieval import retrieve_grounded_context


def test_retrieval_reports_enrichment_usage_metadata(tmp_path: Path) -> None:
    repo, index_dir = _index_repo_with_api_file(tmp_path)
    snapshot = _add_api_enrichment(index_dir)

    response = retrieve_grounded_context(
        snapshot=snapshot,
        query="What part lets another program talk to this tool?",
        limit=1,
        min_score=0.0,
        minimum_results=1,
        minimum_top_score=0.0,
    )

    assert response.is_sufficient is True
    assert response.used_enrichment is True
    assert response.matched_enrichment_terms == [
        "what lets another program talk to this tool",
    ]
    assert response.results[0].chunk.relative_path == "src/code_context/api.py"
    assert "Index enrichment metadata" not in response.results[0].chunk.content


def test_api_ask_reports_enrichment_usage_metadata(tmp_path: Path) -> None:
    repo, index_dir = _index_repo_with_api_file(tmp_path)
    _add_api_enrichment(index_dir)
    client = TestClient(create_app())

    response = client.post(
        "/ask",
        json={
            "index_dir": str(index_dir),
            "question": "What part lets another program talk to this tool?",
            "limit": 1,
            "min_score": 0.0,
            "minimum_results": 1,
            "minimum_top_score": 0.0,
        },
    )

    body = response.json()

    assert response.status_code == 200
    assert body["confidence"] == "grounded"
    assert body["used_enrichment"] is True
    assert body["matched_enrichment_terms"] == [
        "what lets another program talk to this tool",
    ]
    assert body["sources"][0]["relative_path"] == "src/code_context/api.py"


def test_cli_ask_prints_enrichment_usage_metadata(
    capsys,
    tmp_path: Path,
) -> None:
    repo, index_dir = _index_repo_with_api_file(tmp_path)
    _add_api_enrichment(index_dir)

    exit_code = cli.main(
        [
            "ask",
            "--index-dir",
            str(index_dir),
            "--question",
            "What part lets another program talk to this tool?",
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
    assert "Used enrichment: yes" in captured.out
    assert (
        "Matched enrichment terms: what lets another program talk to this tool"
        in captured.out
    )
    assert "src/code_context/api.py" in captured.out
    assert captured.err == ""


def _index_repo_with_api_file(tmp_path: Path) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    api_file = repo / "src" / "code_context" / "api.py"
    api_file.parent.mkdir(parents=True)
    api_file.write_bytes(
        b"def create_app():\n"
        b"    app = object()\n"
        b"    return app\n"
    )

    index_dir = tmp_path / ".code_context_index"
    index_repository(
        repo,
        index_dir=index_dir,
        max_lines=20,
        overlap_lines=0,
    )

    return repo, index_dir


def _add_api_enrichment(index_dir: Path):
    store = JsonIndexStore(index_dir)
    snapshot = store.load()
    api_file = next(
        file
        for file in snapshot.files
        if file.relative_path == "src/code_context/api.py"
    )
    enrichment = FileEnrichment(
        relative_path="src/code_context/api.py",
        source_hash=api_file.content_hash,
        enriched_at=456.78,
        provider="fake",
        model="fake-model",
        summary="Defines the local HTTP API boundary.",
        conceptual_terms=["HTTP API", "external interface"],
        related_user_phrases=[
            "what lets another program talk to this tool",
        ],
        owned_behaviors=["creates API application"],
        important_symbols=["create_app"],
    )
    enriched_snapshot = snapshot.model_copy(update={"enrichments": [enrichment]})
    store.save(enriched_snapshot)
    return enriched_snapshot
