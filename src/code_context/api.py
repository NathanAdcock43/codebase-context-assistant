from __future__ import annotations

"""
ROLE: Expose local codebase indexing, search, and drift detection through FastAPI endpoints.
LAYER: api
FLOW: local_http_api

INPUTS:
- HTTP health requests
- repository indexing requests
- indexed chunk search requests
- drift detection requests

OUTPUTS:
- health responses
- index summary responses
- search result responses with grounded file and line references
- drift report responses

UPSTREAM:
- local developer HTTP clients
- future CLI server command
- future demo scripts
- future LangGraph workflow callers

DOWNSTREAM:
- repository indexing pipeline
- JSON index store
- repository scanner
- drift detector
- local vector store

OWNS:
- FastAPI application creation
- request and response DTOs for HTTP routes
- endpoint orchestration
- HTTP error handling for missing indexes
- local API route definitions

DOES_NOT_OWN:
- repository traversal
- file hashing
- source chunking
- JSON persistence internals
- drift comparison logic
- vector scoring logic
- LLM prompting
- agent workflow behavior

SIDE_EFFECTS:
- /index reads repository files and writes a local JSON index
- /search reads a local JSON index
- /drift reads repository files and a local JSON index

STATE:
  reads:
    - local repository files
    - local JSON index file
  writes:
    - local JSON index file through /index

NOTES:
- Keep endpoints thin and delegate core behavior to existing modules.
- This API exposes grounded context retrieval, not generated answers yet.
- Agent and LLM behavior should be added after these deterministic routes are stable.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from code_context.drift import detect_drift
from code_context.index_store import DEFAULT_INDEX_FILENAME, JsonIndexStore
from code_context.pipeline import index_repository
from code_context.scanner import scan_repository
from code_context.vector_store import search_chunks


class HealthResponse(BaseModel):
    status: str


class IndexRequest(BaseModel):
    repo_path: str
    index_dir: str
    max_lines: int = Field(default=80, ge=1)
    overlap_lines: int = Field(default=10, ge=0)
    index_filename: str = Field(default=DEFAULT_INDEX_FILENAME)


class IndexResponse(BaseModel):
    repo_root: str
    index_path: str
    indexed_at: float
    file_count: int
    chunk_count: int


class SearchRequest(BaseModel):
    index_dir: str
    query: str
    limit: int = Field(default=5, ge=1)
    min_score: float = Field(default=0.0, ge=0.0)
    index_filename: str = Field(default=DEFAULT_INDEX_FILENAME)


class SearchResultResponse(BaseModel):
    chunk_id: str
    relative_path: str
    start_line: int
    end_line: int
    language: str
    score: float
    content: str


class SearchResponse(BaseModel):
    query: str
    result_count: int
    results: list[SearchResultResponse]


class DriftRequest(BaseModel):
    repo_path: str
    index_dir: str
    index_filename: str = Field(default=DEFAULT_INDEX_FILENAME)


def create_app() -> FastAPI:
    """Create and configure the FastAPI app."""
    app = FastAPI(
        title="AI Codebase Context Assistant",
        version="0.1.0",
        description="Local API for indexing a codebase and retrieving grounded source context.",
    )

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok")

    @app.post("/index", response_model=IndexResponse)
    def index_codebase(request: IndexRequest) -> IndexResponse:
        try:
            snapshot = index_repository(
                request.repo_path,
                index_dir=request.index_dir,
                max_lines=request.max_lines,
                overlap_lines=request.overlap_lines,
                index_filename=request.index_filename,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except NotADirectoryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        index_path = JsonIndexStore(
            request.index_dir,
            index_filename=request.index_filename,
        ).index_path

        return IndexResponse(
            repo_root=snapshot.repo_root,
            index_path=str(index_path),
            indexed_at=snapshot.indexed_at,
            file_count=len(snapshot.files),
            chunk_count=len(snapshot.chunks),
        )

    @app.post("/search", response_model=SearchResponse)
    def search_index(request: SearchRequest) -> SearchResponse:
        snapshot = _load_snapshot_or_404(
            request.index_dir,
            index_filename=request.index_filename,
        )

        results = search_chunks(
            snapshot.chunks,
            request.query,
            limit=request.limit,
            min_score=request.min_score,
        )

        response_results = [
            SearchResultResponse(
                chunk_id=result.chunk.chunk_id,
                relative_path=result.chunk.relative_path,
                start_line=result.chunk.start_line,
                end_line=result.chunk.end_line,
                language=result.chunk.language,
                score=result.score,
                content=result.chunk.content,
            )
            for result in results
        ]

        return SearchResponse(
            query=request.query,
            result_count=len(response_results),
            results=response_results,
        )

    @app.post("/drift")
    def drift(request: DriftRequest) -> dict:
        snapshot = _load_snapshot_or_404(
            request.index_dir,
            index_filename=request.index_filename,
        )

        try:
            current_files = scan_repository(Path(request.repo_path))
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except NotADirectoryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        report = detect_drift(snapshot=snapshot, current_files=current_files)
        return report.model_dump()

    return app


def _load_snapshot_or_404(index_dir: str, *, index_filename: str) -> object:
    store = JsonIndexStore(index_dir, index_filename=index_filename)

    try:
        return store.load()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


app = create_app()