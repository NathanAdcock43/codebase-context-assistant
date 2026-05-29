from __future__ import annotations

"""
ROLE: Verify grounded retrieval service behavior and sufficiency checks.
LAYER: tests
FLOW: retrieval_validation

INPUTS:
- IndexSnapshot records
- sample SourceChunk records
- query text
- deterministic enriched retrieval query text
- path-like query fragments
- retrieval thresholds
- temporary index directories

OUTPUTS:
- retrieval response behavior assertions
- enriched retrieval query assertions
- generic ticket anchor retrieval assertions
- weak trailing result pruning assertions
- requested source path sufficiency assertions

UPSTREAM:
- retrieval service implementation
- vector store implementation
- index store implementation
- IndexSnapshot model
- SourceChunk model

DOWNSTREAM:
- local test runs
- future CI guardrails
- FastAPI retrieve endpoint
- future verifier node
- future responder node
- system map review

OWNS:
- retrieval sufficiency tests
- retrieval query enrichment tests
- generic ticket-style anchor retrieval tests
- weak trailing result pruning tests
- requested source path sufficiency tests
- insufficient context tests
- empty query tests
- threshold validation tests
- loaded snapshot retrieval tests

DOES_NOT_OWN:
- scanner tests
- chunker tests
- metadata store tests
- drift detection tests
- API route tests
- LLM answer tests

SIDE_EFFECTS:
- writes temporary JSON index files through pytest tmp_path

STATE:
  reads:
    - temporary JSON index files
  writes:
    - temporary JSON index files

NOTES:
- These tests protect refusal behavior before adding LLM answer generation.
- The retrieval service returns grounded context but does not generate answers.
"""

from pathlib import Path

import pytest

from code_context.index_store import JsonIndexStore, build_index_snapshot
from code_context.models import FileEnrichment, FileMetadata, IndexSnapshot, SourceChunk
from code_context.retrieval import (
    build_searchable_chunks_with_enrichment,
    load_and_retrieve_context,
    retrieve_grounded_context,
)


def test_retrieve_grounded_context_marks_strong_match_as_sufficient() -> None:
    snapshot = _snapshot(
        chunks=[
            _chunk(
                chunk_id="src/scanner.py:1-2:hash",
                relative_path="src/scanner.py",
                content="def calculate_file_hash(path):\n    return hashlib.sha256(data).hexdigest()\n",
            ),
            _chunk(
                chunk_id="src/api.py:1-2:api",
                relative_path="src/api.py",
                content="define FastAPI routes for health checks\n",
            ),
        ]
    )

    response = retrieve_grounded_context(
        snapshot=snapshot,
        query="calculate file hash",
        limit=2,
        minimum_top_score=0.05,
    )

    assert response.is_sufficient is True
    assert response.insufficient_reason is None
    assert response.results[0].chunk.relative_path == "src/scanner.py"
    assert response.results[0].score > 0


def test_retrieve_grounded_context_marks_empty_query_as_insufficient() -> None:
    response = retrieve_grounded_context(
        snapshot=_snapshot(chunks=[_chunk(content="hash content")]),
        query="   ",
    )

    assert response.query == ""
    assert response.is_sufficient is False
    assert response.insufficient_reason == "Query is empty."
    assert response.results == []


def test_retrieve_grounded_context_marks_empty_index_as_insufficient() -> None:
    response = retrieve_grounded_context(
        snapshot=_snapshot(chunks=[]),
        query="hash",
    )

    assert response.query == "hash"
    assert response.is_sufficient is False
    assert response.insufficient_reason == "The index does not contain any chunks to search."
    assert response.results == []


def test_retrieve_grounded_context_marks_no_matching_results_as_insufficient() -> None:
    snapshot = _snapshot(
        chunks=[
            _chunk(
                chunk_id="src/api.py:1-2:api",
                relative_path="src/api.py",
                content="FastAPI route definitions only",
            )
        ]
    )

    response = retrieve_grounded_context(
        snapshot=snapshot,
        query="postgres liquibase migration",
        min_score=0.99,
    )

    assert response.is_sufficient is False
    assert response.insufficient_reason == "Not enough relevant indexed context was found."
    assert response.results == []


def test_retrieve_grounded_context_marks_low_score_result_as_insufficient() -> None:
    snapshot = _snapshot(
        chunks=[
            _chunk(
                chunk_id="src/scanner.py:1-2:hash",
                relative_path="src/scanner.py",
                content="hash route definitions",
            )
        ]
    )

    response = retrieve_grounded_context(
        snapshot=snapshot,
        query="hash",
        min_score=0.0,
        minimum_top_score=0.99,
    )

    assert response.is_sufficient is False
    assert (
        response.insufficient_reason
        == "The best retrieved context was below the sufficiency score threshold."
    )
    assert response.results != []



def test_retrieve_grounded_context_marks_named_missing_source_path_as_insufficient() -> None:
    snapshot = _snapshot(
        chunks=[
            _chunk(
                relative_path="src/code_context/ask.py",
                content="def ask_indexed_code_question():\n    return None",
            ),
            _chunk(
                relative_path="src/code_context/llm_factory.py",
                content="def build_configured_llm_client():\n    return None",
            ),
        ]
    )

    response = retrieve_grounded_context(
        snapshot=snapshot,
        query="In src/code_context/api.py, explain where create_app is defined.",
        limit=2,
        min_score=0.0,
        minimum_results=1,
        minimum_top_score=0.0,
    )

    assert response.is_sufficient is False
    assert response.insufficient_reason == (
        "The query asked about src/code_context/api.py, but retrieved context did not include that file."
    )
    assert [result.chunk.relative_path for result in response.results] == [
        "src/code_context/ask.py",
        "src/code_context/llm_factory.py",
    ]


def test_retrieve_grounded_context_uses_enriched_query_text_for_search(monkeypatch) -> None:
    captured_queries: list[str] = []

    def fake_search_chunks(chunks, query, *, limit, min_score):
        captured_queries.append(query)
        return []

    monkeypatch.setattr("code_context.retrieval.search_chunks", fake_search_chunks)

    query = """
    AMP-14751
    Postgres - Liquibase - BFN_FSCL_YR_END_OPT add new column for CREATE_CLASS_1_4
    Add new column for CREATE_CLASS_1_4 VARCHAR(1) NOT NULL DEFAULT 'N';
    """

    response = retrieve_grounded_context(
        snapshot=_snapshot(chunks=[_chunk(content="unrelated")]),
        query=query,
        limit=3,
        min_score=0.0,
        minimum_results=1,
        minimum_top_score=0.0,
    )

    assert response.query == query.strip()
    assert response.is_sufficient is False
    assert captured_queries
    assert captured_queries[0] != response.query
    assert captured_queries[0].count("BFN_FSCL_YR_END_OPT") > response.query.count("BFN_FSCL_YR_END_OPT")
    assert captured_queries[0].count("CREATE_CLASS_1_4") > response.query.count("CREATE_CLASS_1_4")



def test_retrieve_grounded_context_prunes_weak_trailing_results() -> None:
    snapshot = _snapshot(
        chunks=[
            _chunk(
                chunk_id="src/exports/options.py:1-6",
                relative_path="src/exports/options.py",
                content=(
                    "class ExportRunOptions:\n"
                    "    CUSTOMER_EXPORT_STATUS = 'N'\n"
                    "    def apply_export_status(self):\n"
                    "        return CUSTOMER_EXPORT_STATUS\n"
                ),
            ),
            _chunk(
                chunk_id="src/navigation/menu.py:1-6",
                relative_path="src/navigation/menu.py",
                content=(
                    "class NavigationMenu:\n"
                    "    def render_user_menu(self):\n"
                    "        return 'menu'\n"
                ),
            ),
        ]
    )

    query = """
    TASK-123
    Data change request
    Add CUSTOMER_EXPORT_STATUS to EXPORT_RUN_OPTIONS.
    Users need an option to control export status during run setup.
    """

    response = retrieve_grounded_context(
        snapshot=snapshot,
        query=query,
        limit=2,
        min_score=0.0,
        minimum_results=1,
        minimum_top_score=0.0,
    )

    assert response.is_sufficient is True
    assert [result.chunk.relative_path for result in response.results] == [
        "src/exports/options.py",
    ]

def test_retrieve_grounded_context_uses_enriched_ticket_anchors_for_real_search() -> None:
    snapshot = _snapshot(
        chunks=[
            _chunk(
                chunk_id="src/exports/options.py:1-6",
                relative_path="src/exports/options.py",
                content=(
                    "class ExportRunOptions:\n"
                    "    CUSTOMER_EXPORT_STATUS = 'N'\n"
                    "    def apply_export_status(self):\n"
                    "        return CUSTOMER_EXPORT_STATUS\n"
                ),
            ),
            _chunk(
                chunk_id="src/navigation/menu.py:1-6",
                relative_path="src/navigation/menu.py",
                content=(
                    "class NavigationMenu:\n"
                    "    def render_user_menu(self):\n"
                    "        return 'menu'\n"
                ),
            ),
        ]
    )

    query = """
    TASK-123
    Data change request
    Add CUSTOMER_EXPORT_STATUS to EXPORT_RUN_OPTIONS.
    Users need an option to control export status during run setup.
    """

    response = retrieve_grounded_context(
        snapshot=snapshot,
        query=query,
        limit=2,
        min_score=0.0,
        minimum_results=1,
        minimum_top_score=0.0,
    )

    assert response.is_sufficient is True
    assert response.results[0].chunk.relative_path == "src/exports/options.py"
    assert "CUSTOMER_EXPORT_STATUS" in response.results[0].chunk.content


def test_retrieve_grounded_context_uses_related_terms_for_fuzzy_http_api_question() -> None:
    snapshot = _snapshot(
        chunks=[
            _chunk(
                chunk_id="src/code_context/api.py:1-6",
                relative_path="src/code_context/api.py",
                content=(
                    "def create_app():\n"
                    "    app = FastAPI()\n"
                    "    @app.post('/ask')\n"
                    "    def ask_question():\n"
                    "        return {'answer': 'ok'}\n"
                ),
            ),
            _chunk(
                chunk_id="docs/project_brief.md:1-4",
                relative_path="docs/project_brief.md",
                content=(
                    "This project helps a developer understand a codebase.\n"
                    "It scans files and answers questions with grounded context.\n"
                ),
            ),
        ]
    )

    response = retrieve_grounded_context(
        snapshot=snapshot,
        query="What part lets another program talk to this?",
        related_terms=["HTTP", "API", "FastAPI", "endpoint", "route"],
        limit=2,
        min_score=0.0,
        minimum_results=1,
        minimum_top_score=0.0,
    )

    assert response.query == "What part lets another program talk to this?"
    assert response.related_terms == ["HTTP", "API", "FastAPI", "endpoint", "route"]
    assert response.retrieval_query is not None
    assert "Related terms:" in response.retrieval_query
    assert response.is_sufficient is True
    assert response.results[0].chunk.relative_path == "src/code_context/api.py"



def test_retrieve_grounded_context_uses_fresh_enrichment_metadata_for_fuzzy_query() -> None:
    source_chunk = _chunk(
        chunk_id="src/code_context/api.py:1-3",
        relative_path="src/code_context/api.py",
        content=(
            "def create_app():\n"
            "    app = object()\n"
            "    return app\n"
        ),
    )
    snapshot = build_index_snapshot(
        repo_path="C:/example/repo",
        files=[
            FileMetadata(
                path=Path("C:/example/repo/src/code_context/api.py"),
                relative_path="src/code_context/api.py",
                extension=".py",
                size_bytes=100,
                modified_at=123.45,
                content_hash="fresh-api-hash",
                language="python",
            )
        ],
        chunks=[source_chunk],
        indexed_at=123.45,
    ).model_copy(
        update={
            "enrichments": [
                FileEnrichment(
                    relative_path="src/code_context/api.py",
                    source_hash="fresh-api-hash",
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
            ]
        }
    )

    response = retrieve_grounded_context(
        snapshot=snapshot,
        query="What part lets another program talk to this tool?",
        limit=1,
        min_score=0.0,
        minimum_results=1,
        minimum_top_score=0.0,
    )

    assert response.is_sufficient is True
    assert response.results[0].chunk.relative_path == "src/code_context/api.py"
    assert response.results[0].chunk.content == source_chunk.content
    assert "Index enrichment metadata" not in response.results[0].chunk.content


def test_build_searchable_chunks_with_enrichment_ignores_stale_enrichment_metadata() -> None:
    source_chunk = _chunk(
        chunk_id="src/code_context/api.py:1-3",
        relative_path="src/code_context/api.py",
        content=(
            "def create_app():\\n"
            "    app = object()\\n"
            "    return app\\n"
        ),
    )
    snapshot = build_index_snapshot(
        repo_path="C:/example/repo",
        files=[
            FileMetadata(
                path=Path("C:/example/repo/src/code_context/api.py"),
                relative_path="src/code_context/api.py",
                extension=".py",
                size_bytes=100,
                modified_at=123.45,
                content_hash="fresh-api-hash",
                language="python",
            )
        ],
        chunks=[source_chunk],
        indexed_at=123.45,
    ).model_copy(
        update={
            "enrichments": [
                FileEnrichment(
                    relative_path="src/code_context/api.py",
                    source_hash="old-api-hash",
                    enriched_at=456.78,
                    provider="fake",
                    model="fake-model",
                    summary="Defines the galactic bridge interface.",
                    conceptual_terms=["galactic bridge"],
                    related_user_phrases=["where is the galactic bridge"],
                )
            ]
        }
    )

    searchable_chunks, original_chunks_by_id = build_searchable_chunks_with_enrichment(snapshot)

    assert searchable_chunks == [source_chunk]
    assert original_chunks_by_id == {}
    assert "galactic bridge" not in searchable_chunks[0].content

def test_load_and_retrieve_context_loads_saved_snapshot(tmp_path: Path) -> None:
    index_dir = tmp_path / ".code_context_index"
    snapshot = _snapshot(
        chunks=[
            _chunk(
                chunk_id="src/drift.py:1-2:drift",
                relative_path="src/drift.py",
                content="detect stale indexed metadata by comparing content hashes",
            )
        ]
    )

    JsonIndexStore(index_dir).save(snapshot)

    response = load_and_retrieve_context(
        index_dir=index_dir,
        query="stale content hashes",
    )

    assert response.is_sufficient is True
    assert response.results[0].chunk.relative_path == "src/drift.py"


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"limit": 0}, "limit must be at least 1"),
        ({"min_score": -0.1}, "min_score must be 0 or greater"),
        ({"minimum_results": 0}, "minimum_results must be at least 1"),
        (
            {"limit": 1, "minimum_results": 2},
            "minimum_results must be less than or equal to limit",
        ),
        ({"minimum_top_score": -0.1}, "minimum_top_score must be 0 or greater"),
    ],
)
def test_retrieve_grounded_context_rejects_invalid_settings(
    kwargs: dict,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        retrieve_grounded_context(
            snapshot=_snapshot(chunks=[_chunk(content="hash content")]),
            query="hash",
            **kwargs,
        )


def _snapshot(*, chunks: list[SourceChunk]) -> IndexSnapshot:
    return build_index_snapshot(
        repo_path="C:/example/repo",
        files=[],
        chunks=chunks,
        indexed_at=123.45,
    )


def _chunk(
    *,
    content: str,
    chunk_id: str = "src/example.py:1-1:abc123",
    relative_path: str = "src/example.py",
) -> SourceChunk:
    return SourceChunk(
        chunk_id=chunk_id,
        relative_path=relative_path,
        start_line=1,
        end_line=max(1, content.count("\n")),
        content=content,
        language="python",
    )