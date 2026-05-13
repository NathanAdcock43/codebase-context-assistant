from __future__ import annotations

"""
ROLE: Verify configured LLM client factory behavior.
LAYER: tests
FLOW: llm_client_factory_validation

INPUTS:
- AppConfig records
- explicit environment mappings
- monkeypatched OpenAI client builder
- supported and unsupported provider names
- missing provider credentials

OUTPUTS:
- configured LLM client factory assertions
- provider normalization assertions
- missing configuration error assertions
- unsupported provider error assertions

UPSTREAM:
- LLM client factory
- AppConfig provider settings
- LLM client boundary
- OpenAI adapter builder

DOWNSTREAM:
- local test runs
- future CI guardrails
- future CLI ask LLM wiring tests
- future API ask LLM wiring tests
- future generated ask answer tests
- system map review

OWNS:
- LLM provider factory tests
- provider normalization tests
- disabled provider behavior tests
- OpenAI provider selection tests
- missing key guardrail tests

DOES_NOT_OWN:
- OpenAI SDK tests
- real provider network tests
- prompt construction tests
- grounded answer generation tests
- ask workflow tests
- API route tests
- CLI route tests

SIDE_EFFECTS:
- monkeypatches OpenAI client builder
- captures in-memory fake client construction

STATE:
  reads:
    - in-memory AppConfig records
    - explicit test environment mappings
  writes:
    - in-memory fake client construction records

NOTES:
- These tests do not call OpenAI.
- This keeps provider selection testable before ask is wired to generated answers.
"""

from typing import Any

import pytest

from code_context.config import AppConfig
from code_context.llm import LlmClientUnavailableError, LlmMessage, LlmRequest
from code_context.llm_factory import (
    MissingLlmConfigurationError,
    UnsupportedLlmProviderError,
    build_configured_llm_client,
    build_llm_client_from_config,
    normalize_llm_provider,
)


class FakeLlmClient:
    def complete(self, request: LlmRequest) -> Any:
        return None


def test_normalize_llm_provider_trims_and_lowercases_provider_names() -> None:
    assert normalize_llm_provider(" OpenAI ") == "openai"


@pytest.mark.parametrize("provider", [None, "", "none", "disabled", "off", " DISABLED "])
def test_normalize_llm_provider_returns_none_for_disabled_values(provider: str | None) -> None:
    assert normalize_llm_provider(provider) is None


def test_build_llm_client_from_config_returns_disabled_client_when_provider_is_disabled() -> None:
    client = build_llm_client_from_config(AppConfig(llm_provider="disabled", openai_api_key=None))

    with pytest.raises(LlmClientUnavailableError):
        client.complete(
            LlmRequest(
                model="test-model",
                messages=[
                    LlmMessage(role="user", content="hello"),
                ],
            )
        )


def test_build_llm_client_from_config_builds_openai_client_with_configured_key(monkeypatch: Any) -> None:
    fake_client = FakeLlmClient()
    captured_api_keys: list[str | None] = []

    def fake_builder(*, api_key: str | None = None) -> FakeLlmClient:
        captured_api_keys.append(api_key)
        return fake_client

    monkeypatch.setattr("code_context.llm_factory.build_openai_llm_client", fake_builder)

    client = build_llm_client_from_config(AppConfig(llm_provider="openai", openai_api_key="test-key"))

    assert client is fake_client
    assert captured_api_keys == ["test-key"]


def test_build_llm_client_from_config_rejects_openai_without_api_key() -> None:
    with pytest.raises(MissingLlmConfigurationError, match="OPENAI_API_KEY is required"):
        build_llm_client_from_config(AppConfig(llm_provider="openai", openai_api_key=None))


def test_build_llm_client_from_config_rejects_unsupported_provider() -> None:
    with pytest.raises(UnsupportedLlmProviderError, match="Unsupported LLM provider"):
        build_llm_client_from_config(AppConfig(llm_provider="anthropic", openai_api_key="test-key"))


def test_build_configured_llm_client_loads_environment_mapping(monkeypatch: Any) -> None:
    fake_client = FakeLlmClient()
    captured_api_keys: list[str | None] = []

    def fake_builder(*, api_key: str | None = None) -> FakeLlmClient:
        captured_api_keys.append(api_key)
        return fake_client

    monkeypatch.setattr("code_context.llm_factory.build_openai_llm_client", fake_builder)

    client = build_configured_llm_client(
        {
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "env-test-key",
        }
    )

    assert client is fake_client
    assert captured_api_keys == ["env-test-key"]
