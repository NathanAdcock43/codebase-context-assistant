from __future__ import annotations

"""
ROLE: Verify provider-neutral LLM client boundary behavior.
LAYER: tests
FLOW: llm_client_boundary_validation

INPUTS:
- sample LLM messages
- sample LLM requests
- sample LLM responses
- invalid model and temperature values
- disabled LLM client calls

OUTPUTS:
- LLM model validation assertions
- disabled client behavior assertions
- usage metadata normalization assertions

UPSTREAM:
- LLM boundary module
- future LLM provider adapter
- future grounded answer generation service
- future responder node LLM integration

DOWNSTREAM:
- local test runs
- future CI guardrails
- future OpenAI client tests
- future grounded answer generation tests
- system map review

OWNS:
- provider-neutral LLM DTO tests
- LLM request validation tests
- LLM response validation tests
- disabled client placeholder tests

DOES_NOT_OWN:
- OpenAI SDK tests
- prompt construction tests
- retrieval tests
- verifier tests
- API route tests
- CLI tests
- LangGraph tests

SIDE_EFFECTS:
- none

STATE:
  reads:
    - in-memory test values
  writes:
    - none

NOTES:
- These tests do not require a real provider key.
- This keeps the LLM seam testable before any LLM call is added.
"""

import pytest

from code_context.llm import (
    DisabledLlmClient,
    LlmClientUnavailableError,
    LlmMessage,
    LlmRequest,
    LlmResponse,
    build_disabled_llm_client,
    build_usage_metadata,
)


def test_llm_message_trims_content() -> None:
    message = LlmMessage(role="user", content="  Where is the scanner?  ")

    assert message.content == "Where is the scanner?"


def test_llm_request_requires_at_least_one_message() -> None:
    with pytest.raises(ValueError, match="at least one message"):
        LlmRequest(messages=[], model="test-model")


@pytest.mark.parametrize("temperature", [-0.1, 2.1])
def test_llm_request_rejects_temperature_outside_supported_range(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature must be between 0 and 2"):
        LlmRequest(
            messages=[LlmMessage(role="user", content="Question")],
            model="test-model",
            temperature=temperature,
        )


def test_llm_response_uses_independent_usage_defaults() -> None:
    first = LlmResponse(content="Answer", model="test-model", provider="test")
    second = LlmResponse(content="Answer", model="test-model", provider="test")

    first.usage["prompt_tokens"] = 10

    assert second.usage == {}


def test_disabled_llm_client_raises_clear_error() -> None:
    client = DisabledLlmClient()
    request = LlmRequest(
        messages=[LlmMessage(role="user", content="Question")],
        model="test-model",
    )

    with pytest.raises(LlmClientUnavailableError, match="LLM answer generation is not configured yet"):
        client.complete(request)


def test_build_disabled_llm_client_returns_disabled_client_boundary() -> None:
    client = build_disabled_llm_client()

    assert isinstance(client, DisabledLlmClient)


def test_build_usage_metadata_returns_plain_copy() -> None:
    usage = {"prompt_tokens": 10, "completion_tokens": 5}

    result = build_usage_metadata(usage)

    assert result == usage
    assert result is not usage