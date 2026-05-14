from __future__ import annotations

"""
ROLE: Verify grounded answer generation orchestration behavior.
LAYER: tests
FLOW: grounded_answer_generation_validation

INPUTS:
- sample developer questions
- sample SearchResult records
- sample SourceChunk records
- fake LLM clients
- generated LLM response records
- result limit settings

OUTPUTS:
- generated answer assertions
- LLM request capture assertions
- grounded source reference assertions
- validation error assertions

UPSTREAM:
- grounded answer generation service
- prompt builder module
- LLM client boundary
- SearchResult model
- SourceChunk model

DOWNSTREAM:
- local test runs
- future CI guardrails
- future OpenAI provider tests
- future responder node LLM integration tests
- future API and CLI ask integration tests
- system map review

OWNS:
- answer generation service tests
- injected client invocation tests
- generated answer shape tests
- grounded source reference tests
- source limit validation tests

DOES_NOT_OWN:
- OpenAI SDK tests
- prompt formatting unit tests
- retrieval tests
- verifier tests
- stale-index refusal tests
- API route tests
- CLI tests
- LangGraph tests

SIDE_EFFECTS:
- fake clients capture in-memory LLM requests

STATE:
  reads:
    - in-memory test values
  writes:
    - in-memory fake client request capture

NOTES:
- These tests do not call a real LLM provider.
- This protects the seam where future generated answers will be added after verification.
"""

import pytest

from code_context.answer_generation import (
    GeneratedAnswer,
    GroundedSourceReference,
    build_grounded_source_references,
    generate_grounded_answer,
)
from code_context.llm import LlmClientUnavailableError, LlmRequest, LlmResponse
from code_context.models import SearchResult, SourceChunk


class FakeLlmClient:
    def __init__(self, response: LlmResponse) -> None:
        self.response = response
        self.requests: list[LlmRequest] = []

    def complete(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        return self.response


class FailingLlmClient:
    def complete(self, request: LlmRequest) -> LlmResponse:
        raise LlmClientUnavailableError("client unavailable")


def test_generate_grounded_answer_calls_injected_client_and_returns_answer() -> None:
    client = FakeLlmClient(
        LlmResponse(
            content="The scanner is implemented in src/code_context/scanner.py lines 10-20.",
            model="test-model",
            provider="test-provider",
            usage={"prompt_tokens": 10, "completion_tokens": 8},
        )
    )

    answer = generate_grounded_answer(
        question="Where is scanning implemented?",
        results=[
            _result(
                relative_path="src/code_context/scanner.py",
                start_line=10,
                end_line=20,
                content="def scan_repository():\n    return []",
            )
        ],
        client=client,
        model="test-model",
    )

    assert answer.answer == "The scanner is implemented in src/code_context/scanner.py lines 10-20."
    assert answer.model == "test-model"
    assert answer.provider == "test-provider"
    assert answer.usage == {"prompt_tokens": 10, "completion_tokens": 8}
    assert len(answer.sources) == 1
    assert answer.sources[0].relative_path == "src/code_context/scanner.py"
    assert client.requests


def test_generate_grounded_answer_omits_temperature_by_default() -> None:
    client = FakeLlmClient(
        LlmResponse(
            content="Answer from grounded context.",
            model="test-model",
            provider="test-provider",
        )
    )

    generate_grounded_answer(
        question="Where is the API created?",
        results=[
            _result(
                relative_path="src/code_context/api.py",
                start_line=15,
                end_line=30,
                content="def create_app():\n    return app",
            )
        ],
        client=client,
        model="test-model",
    )

    request = client.requests[0]

    assert request.model == "test-model"
    assert request.temperature is None


def test_generate_grounded_answer_builds_grounded_prompt_request() -> None:
    client = FakeLlmClient(
        LlmResponse(
            content="Answer from grounded context.",
            model="test-model",
            provider="test-provider",
        )
    )

    generate_grounded_answer(
        question="Where is the API created?",
        results=[
            _result(
                relative_path="src/code_context/api.py",
                start_line=15,
                end_line=30,
                content="def create_app():\n    return app",
            )
        ],
        client=client,
        model="test-model",
        temperature=0.1,
    )

    request = client.requests[0]

    assert request.model == "test-model"
    assert request.temperature == 0.1
    assert len(request.messages) == 2
    assert request.messages[0].role == "system"
    assert request.messages[1].role == "user"
    assert "Where is the API created?" in request.messages[1].content
    assert "src/code_context/api.py" in request.messages[1].content
    assert "Lines: 15-30" in request.messages[1].content


def test_generate_grounded_answer_limits_sources_sent_to_client_and_result() -> None:
    client = FakeLlmClient(
        LlmResponse(
            content="Answer from first source.",
            model="test-model",
            provider="test-provider",
        )
    )

    answer = generate_grounded_answer(
        question="Where is configuration handled?",
        results=[
            _result(relative_path="first.py", start_line=1, end_line=2, content="first"),
            _result(relative_path="second.py", start_line=3, end_line=4, content="second"),
        ],
        client=client,
        model="test-model",
        max_results=1,
    )

    prompt = client.requests[0].messages[1].content

    assert "first.py" in prompt
    assert "second.py" not in prompt
    assert [source.relative_path for source in answer.sources] == ["first.py"]


def test_build_grounded_source_references_maps_search_results() -> None:
    references = build_grounded_source_references(
        [
            _result(
                relative_path="src/code_context/drift.py",
                start_line=5,
                end_line=12,
                score=0.81,
            )
        ]
    )

    assert references == [
        GroundedSourceReference(
            relative_path="src/code_context/drift.py",
            start_line=5,
            end_line=12,
            score=0.81,
        )
    ]


def test_generate_grounded_answer_rejects_empty_results() -> None:
    client = FakeLlmClient(
        LlmResponse(
            content="Unused",
            model="test-model",
            provider="test-provider",
        )
    )

    with pytest.raises(ValueError, match="At least one grounded search result is required"):
        generate_grounded_answer(
            question="Where is scanning?",
            results=[],
            client=client,
            model="test-model",
        )


def test_generate_grounded_answer_rejects_invalid_max_results() -> None:
    client = FakeLlmClient(
        LlmResponse(
            content="Unused",
            model="test-model",
            provider="test-provider",
        )
    )

    with pytest.raises(ValueError, match="Maximum result count must be at least 1"):
        generate_grounded_answer(
            question="Where is scanning?",
            results=[_result()],
            client=client,
            model="test-model",
            max_results=0,
        )


def test_generate_grounded_answer_propagates_client_unavailable_error() -> None:
    with pytest.raises(LlmClientUnavailableError, match="client unavailable"):
        generate_grounded_answer(
            question="Where is scanning?",
            results=[_result()],
            client=FailingLlmClient(),
            model="test-model",
        )


def test_generated_answer_requires_sources() -> None:
    with pytest.raises(ValueError, match="at least one grounded source"):
        GeneratedAnswer(
            answer="Answer",
            model="test-model",
            provider="test-provider",
            sources=[],
        )


def _result(
    *,
    relative_path: str = "src/code_context/scanner.py",
    start_line: int = 1,
    end_line: int = 3,
    content: str = "def scan_repository():\n    return []",
    language: str = "python",
    score: float = 0.75,
) -> SearchResult:
    return SearchResult(
        chunk=SourceChunk(
            chunk_id=f"{relative_path}:{start_line}-{end_line}",
            relative_path=relative_path,
            start_line=start_line,
            end_line=end_line,
            content=content,
            language=language,
        ),
        score=score,
    )