from __future__ import annotations

"""
ROLE: Generate grounded answer records through an injected LLM client boundary.
LAYER: application_service
FLOW: grounded_answer_generation

INPUTS:
- developer question text
- verified grounded SearchResult records
- provider-neutral LlmClient implementations
- LLM model names
- optional prompt temperature settings
- source result limits
- source content character limits

OUTPUTS:
- GeneratedAnswer records
- GroundedSourceReference records
- provider-neutral LLM request execution through injected clients

UPSTREAM:
- prompt builder module
- LLM client boundary
- retrieval service
- verifier node
- future responder node LLM integration
- local tests

DOWNSTREAM:
- future ask workflow LLM integration
- future FastAPI ask responses
- future CLI ask responses
- future OpenAI provider adapter
- future citation verification

OWNS:
- grounded answer generation orchestration
- prompt builder invocation
- injected LLM client invocation
- generated answer result shape
- grounded source reference extraction
- source limit validation

DOES_NOT_OWN:
- OpenAI SDK calls
- LLM provider configuration loading
- retrieval scoring
- retrieval sufficiency decisions
- stale-index refusal
- prompt formatting internals
- API routing
- CLI argument parsing
- LangGraph workflow routing

SIDE_EFFECTS:
- calls the injected LlmClient.complete implementation

STATE:
  reads:
    - in-memory SearchResult records
  writes:
    - none

NOTES:
- This module does not create a real provider client.
- This module should only be called after stale-index and sufficiency checks pass.
- Temperature is optional because some provider models reject sampling parameters.
- Tests use fake clients so no provider key or network call is required.
"""

from collections.abc import Sequence

from pydantic import BaseModel, Field, field_validator

from code_context.llm import LlmClient
from code_context.models import SearchResult
from code_context.prompts import build_grounded_answer_request


class GroundedSourceReference(BaseModel):
    """Source reference attached to a generated grounded answer."""

    relative_path: str
    start_line: int
    end_line: int
    score: float

    @field_validator("relative_path")
    @classmethod
    def _relative_path_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            msg = "Source relative path cannot be empty."
            raise ValueError(msg)
        return normalized

    @field_validator("start_line", "end_line")
    @classmethod
    def _line_number_must_be_positive(cls, value: int) -> int:
        if value < 1:
            msg = "Source line numbers must be positive."
            raise ValueError(msg)
        return value


class GeneratedAnswer(BaseModel):
    """Generated answer plus grounding metadata."""

    answer: str
    model: str
    provider: str
    sources: list[GroundedSourceReference]
    usage: dict[str, int] = Field(default_factory=dict)

    @field_validator("answer", "model", "provider")
    @classmethod
    def _required_text_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            msg = "Generated answer text fields cannot be empty."
            raise ValueError(msg)
        return normalized

    @field_validator("sources")
    @classmethod
    def _sources_must_not_be_empty(cls, value: list[GroundedSourceReference]) -> list[GroundedSourceReference]:
        if not value:
            msg = "Generated answers must include at least one grounded source."
            raise ValueError(msg)
        return value


def generate_grounded_answer(
    *,
    question: str,
    results: Sequence[SearchResult],
    client: LlmClient,
    model: str,
    temperature: float | None = None,
    max_results: int = 5,
    max_source_chars_per_result: int = 1200,
) -> GeneratedAnswer:
    """Generate an answer from verified grounded retrieval results."""

    selected_results = _select_results(results, max_results=max_results)

    request = build_grounded_answer_request(
        question=question,
        results=selected_results,
        model=model,
        temperature=temperature,
        max_results=max_results,
        max_source_chars_per_result=max_source_chars_per_result,
    )

    response = client.complete(request)

    return GeneratedAnswer(
        answer=response.content,
        model=response.model,
        provider=response.provider,
        sources=build_grounded_source_references(selected_results),
        usage=response.usage,
    )


def build_grounded_source_references(results: Sequence[SearchResult]) -> list[GroundedSourceReference]:
    """Build source references from retrieval results."""

    if not results:
        msg = "At least one grounded search result is required."
        raise ValueError(msg)

    return [
        GroundedSourceReference(
            relative_path=result.chunk.relative_path,
            start_line=result.chunk.start_line,
            end_line=result.chunk.end_line,
            score=result.score,
        )
        for result in results
    ]


def _select_results(results: Sequence[SearchResult], *, max_results: int) -> list[SearchResult]:
    if max_results < 1:
        msg = "Maximum result count must be at least 1."
        raise ValueError(msg)

    if not results:
        msg = "At least one grounded search result is required."
        raise ValueError(msg)

    return list(results[:max_results])