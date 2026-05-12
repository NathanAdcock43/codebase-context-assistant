from __future__ import annotations

"""
ROLE: Expose local codebase indexing, search, retrieval, ask, and drift detection through FastAPI endpoints.
LAYER: api
FLOW: local_http_api

INPUTS:
- HTTP health requests
- repository indexing requests
- indexed chunk search requests
- grounded retrieval requests
- deterministic ask requests
- drift detection requests

OUTPUTS:
- health responses
- index summary responses
- search result responses with grounded file and line references
- retrieval responses with sufficiency decisions
- ask responses with deterministic grounded answers or refusals
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
- retrieval service
- deterministic agent workflow runner
- drift detector
- local vector store

OWNS:
- FastAPI application creation
- request and response DTOs for HTTP routes
- endpoint orchestration
- HTTP error handling for missing indexes
- local API route definitions
- API response shaping for deterministic ask results

DOES_NOT_OWN:
- repository traversal
- file hashing
- source chunking
- JSON persistence internals
- retrieval sufficiency logic
- drift comparison logic
- vector scoring logic
- deterministic agent workflow behavior
- LLM prompting

SIDE_EFFECTS:
- /index reads repository files and writes a local JSON index
- /search reads a local JSON index
- /retrieve reads a local JSON index
- /ask reads a local JSON index
- /drift reads repository files and a local JSON index

STATE:
  reads:
    - local repository files
    - local JSON index file
  writes:
    - local JSON index file through /index

NOTES:
- Keep endpoints thin and delegate core behavior to existing modules.
- This API exposes deterministic grounded workflow answers, not LLM-generated answers yet.
- LangGraph and LLM behavior should be added after these deterministic routes are stable.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ValidationError

from code_context.agent.graph import run_code_question_workflow
from code_context.drift import detect_drift
from code_context.index_store import DEFAULT_INDEX_FILENAME, JsonIndexStore
from code_context.models import IndexSnapshot, SearchResult
from code_context.pipeline import index_repository
from code_context.retrieval import (
    DEFAULT_MINIMUM_TOP_SCORE,
    DEFAULT_RETRIEVAL_MIN_SCORE,
    load_and_retrieve_context,
)
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


class RetrieveRequest(BaseModel):
    index_dir: str
    query: str
    limit: int = Field(default=5, ge=1)
    min_score: float = Field(default=DEFAULT_RETRIEVAL_MIN_SCORE, ge=0.0)
    minimum_results: int = Field(default=1, ge=1)
    minimum_top_score: float = Field(default=DEFAULT_MINIMUM_TOP_SCORE, ge=0.0)
    index_filename: str = Field(default=DEFAULT_INDEX_FILENAME)


class RetrieveResponse(BaseModel):
    query: str
    is_sufficient: bool
    insufficient_reason: str | None
    result_count: int
    results: list[SearchResultResponse]


class AskRequest(BaseModel):
    index_dir: str
    question: str = Field(min_length=1)
    limit: int = Field(default=5, ge=1)
    min_score: float = Field(default=DEFAULT_RETRIEVAL_MIN_SCORE, ge=0.0)
    minimum_results: int = Field(default=1, ge=1)
    minimum_top_score: float = Field(default=DEFAULT_MINIMUM_TOP_SCORE, ge=0.0)
    index_filename: str = Field(default=DEFAULT_INDEX_FILENAME)


class AskSourceResponse(BaseModel):
    chunk_id: str
    relative_path: str
    start_line: int
    end_line: int
    language: str
    score: float


class AskStepResponse(BaseModel):
    name: str
    status: str
    notes: list[str]


class AskResponse(BaseModel):
    question: str
    answer: str
    confidence: str
    is_grounded: bool
    insufficient_reason: str | None
    plan: list[str]
    citations: list[str]
    sources: list[AskSourceResponse]
    steps: list[AskStepResponse]


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

        response_results = _to_search_result_responses(results)

        return SearchResponse(
            query=request.query,
            result_count=len(response_results),
            results=response_results,
        )

    @app.post("/retrieve", response_model=RetrieveResponse)
    def retrieve_context(request: RetrieveRequest) -> RetrieveResponse:
        try:
            retrieval_response = load_and_retrieve_context(
                index_dir=request.index_dir,
                query=request.query,
                limit=request.limit,
                min_score=request.min_score,
                minimum_results=request.minimum_results,
                minimum_top_score=request.minimum_top_score,
                index_filename=request.index_filename,
            )
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        response_results = _to_search_result_responses(retrieval_response.results)

        return RetrieveResponse(
            query=retrieval_response.query,
            is_sufficient=retrieval_response.is_sufficient,
            insufficient_reason=retrieval_response.insufficient_reason,
            result_count=len(response_results),
            results=response_results,
        )

    @app.post("/ask", response_model=AskResponse)
    def ask_question(request: AskRequest) -> AskResponse:
        snapshot = _load_snapshot_or_404(
            request.index_dir,
            index_filename=request.index_filename,
        )

        try:
            state = run_code_question_workflow(
                question=request.question,
                snapshot=snapshot,
                limit=request.limit,
                min_score=request.min_score,
                minimum_results=request.minimum_results,
                minimum_top_score=request.minimum_top_score,
            )
        except ValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        verification = state.verification
        is_grounded = bool(verification and verification.can_answer and verification.is_grounded)
        confidence = "grounded" if is_grounded else "insufficient_context"
        insufficient_reason = None if is_grounded else verification.reason if verification else None
        results = state.retrieval.results if state.retrieval else []

        return AskResponse(
            question=state.question,
            answer=state.answer or "",
            confidence=confidence,
            is_grounded=is_grounded,
            insufficient_reason=insufficient_reason,
            plan=state.plan,
            citations=state.citations,
            sources=_to_ask_source_responses(results),
            steps=[
                AskStepResponse(
                    name=step.name,
                    status=step.status.value,
                    notes=step.notes,
                )
                for step in state.steps
            ],
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


def _load_snapshot_or_404(index_dir: str, *, index_filename: str) -> IndexSnapshot:
    store = JsonIndexStore(index_dir, index_filename=index_filename)

    try:
        return store.load()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _to_search_result_responses(results: list[SearchResult]) -> list[SearchResultResponse]:
    return [
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


def _to_ask_source_responses(results: list[SearchResult]) -> list[AskSourceResponse]:
    return [
        AskSourceResponse(
            chunk_id=result.chunk.chunk_id,
            relative_path=result.chunk.relative_path,
            start_line=result.chunk.start_line,
            end_line=result.chunk.end_line,
            language=result.chunk.language,
            score=result.score,
        )
        for result in results
    ]


app = create_app()