from __future__ import annotations

"""
ROLE: Verify the manual OpenAI smoke-test script without making provider calls.
LAYER: tests
FLOW: openai_connection_smoke_test_validation

INPUTS:
- monkeypatched environment variables
- monkeypatched OpenAI client builder
- fake LLM client responses
- captured terminal output

OUTPUTS:
- smoke script behavior assertions
- missing-key guardrail assertions
- successful fake-client output assertions

UPSTREAM:
- OpenAI smoke-test script
- OpenAI client adapter
- LLM client boundary
- future LLM wiring setup

DOWNSTREAM:
- local test runs
- future CI guardrails
- system map review
- future README setup validation

OWNS:
- smoke script unit tests
- environment guardrail tests
- fake client success-path tests
- no-network smoke-script coverage

DOES_NOT_OWN:
- real OpenAI API calls
- ask workflow tests
- API route tests
- CLI ask tests
- retrieval tests
- stale-index refusal tests

SIDE_EFFECTS:
- monkeypatches environment variables
- monkeypatches smoke script client builder
- captures stdout and stderr

STATE:
  reads:
    - in-memory monkeypatched environment
  writes:
    - captured stdout
    - captured stderr

NOTES:
- These tests do not call OpenAI.
- The real smoke test is run manually from the terminal after OPENAI_API_KEY is loaded.
"""

from typing import Any

from code_context.llm import LlmRequest, LlmResponse
from code_context.scripts import smoke_openai_connection


class FakeLlmClient:
    def __init__(self) -> None:
        self.requests: list[LlmRequest] = []

    def complete(self, request: LlmRequest) -> LlmResponse:
        self.requests.append(request)
        return LlmResponse(
            content="The OpenAI connection works.",
            model=request.model,
            provider="openai",
            usage={"input_tokens": 12, "output_tokens": 6},
        )


def test_smoke_script_returns_error_when_api_key_is_missing(monkeypatch: Any, capsys: Any) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    exit_code = smoke_openai_connection.main()

    captured = capsys.readouterr()

    assert exit_code == 1
    assert "OPENAI_API_KEY is not set" in captured.err


def test_smoke_script_calls_fake_client_when_key_is_set(monkeypatch: Any, capsys: Any) -> None:
    fake_client = FakeLlmClient()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    monkeypatch.setattr(smoke_openai_connection, "build_openai_llm_client", lambda *, api_key=None: fake_client)

    exit_code = smoke_openai_connection.main()

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "OpenAI connection smoke test succeeded." in captured.out
    assert "Provider: openai" in captured.out
    assert "Model: test-model" in captured.out
    assert "Answer: The OpenAI connection works." in captured.out
    assert "Usage:" in captured.out
    assert len(fake_client.requests) == 1
    assert fake_client.requests[0].model == "test-model"


def test_smoke_script_uses_config_default_model_when_openai_model_is_not_set(monkeypatch: Any, capsys: Any) -> None:
    fake_client = FakeLlmClient()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.setattr(smoke_openai_connection, "build_openai_llm_client", lambda *, api_key=None: fake_client)

    exit_code = smoke_openai_connection.main()

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Model: gpt-4.1-mini" in captured.out
    assert fake_client.requests[0].model == "gpt-4.1-mini"
