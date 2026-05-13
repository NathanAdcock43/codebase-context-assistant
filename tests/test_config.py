from __future__ import annotations

"""
ROLE: Verify environment-backed application configuration behavior.
LAYER: tests
FLOW: configuration_validation

INPUTS:
- explicit test environment mappings
- default AppConfig values
- invalid API port values

OUTPUTS:
- configuration model behavior assertions
- environment parsing assertions
- validation error assertions

UPSTREAM:
- config module
- .env.example documentation
- future CLI configuration wiring
- future API configuration wiring
- future LLM provider configuration

DOWNSTREAM:
- local test runs
- future CI guardrails
- future LLM integration tests
- future API startup configuration tests
- system map review

OWNS:
- AppConfig default tests
- environment override tests
- optional empty secret normalization tests
- API port validation tests

DOES_NOT_OWN:
- CLI behavior tests
- API route tests
- LLM provider client tests
- embedding provider tests
- index store tests
- vector store tests

SIDE_EFFECTS:
- none

STATE:
  reads:
    - in-memory environment mappings
  writes:
    - none

NOTES:
- These tests keep configuration deterministic by passing explicit environment mappings.
- Do not require a real .env file or real provider key.
"""

from pathlib import Path

import pytest

from code_context.config import AppConfig, load_app_config


def test_app_config_defaults_match_local_development_expectations() -> None:
    config = AppConfig()

    assert config.environment == "local"
    assert config.index_dir == Path(".code_context_index")
    assert config.index_filename == "code_context_index.json"
    assert config.default_encoding == "utf-8"
    assert config.llm_provider == "openai"
    assert config.openai_api_key is None
    assert config.openai_model == "gpt-4.1-mini"
    assert config.embedding_provider == "local"
    assert config.embedding_model is None
    assert config.chroma_persist_dir == Path(".chroma")
    assert config.api_host == "127.0.0.1"
    assert config.api_port == 8000


def test_load_app_config_uses_explicit_environment_mapping() -> None:
    config = load_app_config(
        {
            "CODE_CONTEXT_ENV": "test",
            "CODE_CONTEXT_INDEX_DIR": ".custom_index",
            "CODE_CONTEXT_INDEX_FILENAME": "custom_index.json",
            "CODE_CONTEXT_DEFAULT_ENCODING": "utf-8-sig",
            "LLM_PROVIDER": "openai",
            "OPENAI_API_KEY": "test-key",
            "OPENAI_MODEL": "test-model",
            "EMBEDDING_PROVIDER": "openai",
            "EMBEDDING_MODEL": "test-embedding",
            "CHROMA_PERSIST_DIR": ".custom_chroma",
            "API_HOST": "0.0.0.0",
            "API_PORT": "9000",
        }
    )

    assert config.environment == "test"
    assert config.index_dir == Path(".custom_index")
    assert config.index_filename == "custom_index.json"
    assert config.default_encoding == "utf-8-sig"
    assert config.openai_api_key == "test-key"
    assert config.openai_model == "test-model"
    assert config.embedding_provider == "openai"
    assert config.embedding_model == "test-embedding"
    assert config.chroma_persist_dir == Path(".custom_chroma")
    assert config.api_host == "0.0.0.0"
    assert config.api_port == 9000


def test_load_app_config_treats_blank_optional_provider_values_as_none() -> None:
    config = load_app_config(
        {
            "OPENAI_API_KEY": "   ",
            "EMBEDDING_MODEL": "",
        }
    )

    assert config.openai_api_key is None
    assert config.embedding_model is None


def test_load_app_config_rejects_invalid_integer_port() -> None:
    with pytest.raises(ValueError, match="API_PORT must be an integer"):
        load_app_config({"API_PORT": "not-a-number"})


@pytest.mark.parametrize("port", ["0", "65536"])
def test_load_app_config_rejects_out_of_range_api_port(port: str) -> None:
    with pytest.raises(ValueError, match="API port must be between 1 and 65535"):
        load_app_config({"API_PORT": port})