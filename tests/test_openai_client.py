from __future__ import annotations

"""
ROLE: Verify optional OpenAI LLM adapter behavior.
LAYER: tests
FLOW: openai_llm_adapter_validation

INPUTS:
- provider-neutral LlmRequest records
- fake OpenAI SDK client objects
- fake OpenAI Responses API response objects
- fake usage metadata objects
- optional client factories

OUTPUTS:
- OpenAI adapter behavior assertions
- provider-neutral LlmResponse assertions
- request conversion assertions
- response extraction assertions
- unavailable-client error assertions

UPSTREAM:
- OpenAI client adapter
- LLM client boundary
- future generated answer service integration
- future provider configuration wiring

DOWNSTREAM:
- local test runs
- future CI guardrails
- future API ask LLM integration tests
- future CLI ask LLM integration tests
- system map review

OWNS:
- OpenAI adapter unit tests
- fake client invocation tests
- OpenAI request shape tests
- OpenAI response conversion tests
- missing SDK guardrail tests

DOES_NOT_OWN:
- OpenAI SDK integration tests
- real network tests
- prompt construction tests
- retrieval tests
- verifier tests
- stale-index refusal tests
- API route tests
- CLI tests

SIDE_EFFECTS:
- fake clients capture in-memory request payloads
- one test monkeypatches Python import behavior

STATE:
  reads:
    - in-memory test values
  writes:
    - in-memory fake client request capture

NOTES:
- These tests do not require the OpenAI package.
- These tests do not call the network.
- Real provider integration should remain behind this adapter.
"""

import builtins
from typing import Any

import pytest

from code_context.llm import LlmMessage, LlmRequest
from code_context.openai_client import (
    OpenAiClientUnavailableError,
    OpenAiLlmClient,
    OpenAiResponseError,
    build_openai_llm_client,
)


class FakeUsageObject:
    input_tokens = 10
    output_tokens = 5
    total_tokens = 15


class FakeOpenAiResponse:
    def __init__(
        self,
        *,
        output_text: str = "Grounded generated answer.",
        model: str | None = "test-response-model",
        usage: Any = None,
    ) -> None:
        self.output_text = output_text
        self.model = model
        self.usage = usage


class FakeResponsesResource:
    def __init__(self, response: FakeOpenAiResponse) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> FakeOpenAiResponse:
        self.calls.append(kwargs)
        return self.response


class FakeOpenAiSdkClient:
    def __init__(self, response: FakeOpenAiResponse) -> None:
        self.responses = FakeResponsesResource(response)


def test_openai_llm_client_calls_responses_api_and_returns_provider_neutral_response() -> None:
    sdk_client = FakeOpenAiSdkClient(
        FakeOpenAiResponse(
            output_text="Use src/code_context/scanner.py lines 10-20.",
            model="test-response-model",
            usage=FakeUsageObject(),
        )
    )
    client = OpenAiLlmClient(sdk_client)

    response = client.complete(_request())

    assert response.content == "Use src/code_context/scanner.py lines 10-20."
    assert response.model == "test-response-model"
    assert response.provider == "openai"
    assert response.usage == {
        "input_tokens": 10,
        "output_tokens": 5,
        "total_tokens": 15,
    }


def test_openai_llm_client_converts_messages_to_openai_input_items() -> None:
    sdk_client = FakeOpenAiSdkClient(FakeOpenAiResponse())
    client = OpenAiLlmClient(sdk_client)

    client.complete(
        LlmRequest(
            messages=[
                LlmMessage(role="system", content="Use only grounded context."),
                LlmMessage(role="user", content="Where is the API created?"),
            ],
            model="test-model",
            temperature=0.1,
        )
    )

    call = sdk_client.responses.calls[0]

    assert call == {
        "model": "test-model",
        "input": [
            {"role": "system", "content": "Use only grounded context."},
            {"role": "user", "content": "Where is the API created?"},
        ],
        "temperature": 0.1,
    }


def test_openai_llm_client_uses_request_model_when_response_model_is_missing() -> None:
    sdk_client = FakeOpenAiSdkClient(FakeOpenAiResponse(model=None))
    client = OpenAiLlmClient(sdk_client)

    response = client.complete(_request(model="fallback-model"))

    assert response.model == "fallback-model"


def test_openai_llm_client_extracts_usage_from_mapping() -> None:
    sdk_client = FakeOpenAiSdkClient(
        FakeOpenAiResponse(
            usage={
                "input_tokens": 3,
                "output_tokens": 4,
                "ignored_text": "not-an-int",
            }
        )
    )
    client = OpenAiLlmClient(sdk_client)

    response = client.complete(_request())

    assert response.usage == {
        "input_tokens": 3,
        "output_tokens": 4,
    }


def test_openai_llm_client_rejects_missing_output_text() -> None:
    sdk_client = FakeOpenAiSdkClient(FakeOpenAiResponse(output_text="   "))
    client = OpenAiLlmClient(sdk_client)

    with pytest.raises(OpenAiResponseError, match="did not include output_text"):
        client.complete(_request())


def test_build_openai_llm_client_uses_injected_client_factory_without_importing_sdk() -> None:
    calls: list[dict[str, str]] = []

    def factory(**kwargs: str) -> FakeOpenAiSdkClient:
        calls.append(kwargs)
        return FakeOpenAiSdkClient(FakeOpenAiResponse())

    client = build_openai_llm_client(api_key="test-key", client_factory=factory)

    assert isinstance(client, OpenAiLlmClient)
    assert calls == [{"api_key": "test-key"}]


def test_build_openai_llm_client_allows_sdk_to_read_environment_when_api_key_is_not_passed() -> None:
    calls: list[dict[str, str]] = []

    def factory(**kwargs: str) -> FakeOpenAiSdkClient:
        calls.append(kwargs)
        return FakeOpenAiSdkClient(FakeOpenAiResponse())

    build_openai_llm_client(client_factory=factory)

    assert calls == [{}]


def test_build_openai_llm_client_raises_clear_error_when_openai_sdk_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "openai":
            raise ImportError("missing openai")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(OpenAiClientUnavailableError, match="OpenAI SDK is not installed"):
        build_openai_llm_client()


def _request(*, model: str = "test-model") -> LlmRequest:
    return LlmRequest(
        messages=[
            LlmMessage(role="system", content="Use only grounded context."),
            LlmMessage(role="user", content="Where is scanning implemented?"),
        ],
        model=model,
    )