from __future__ import annotations

"""
ROLE: Build configured provider-neutral LLM clients from application configuration.
LAYER: llm
FLOW: llm_client_factory

INPUTS:
- AppConfig records
- optional explicit environment mappings
- configured LLM provider names
- OpenAI API key configuration

OUTPUTS:
- provider-neutral LlmClient implementations
- disabled LLM clients when provider use is explicitly disabled
- clear configuration errors for missing or unsupported providers

UPSTREAM:
- AppConfig provider settings
- local environment variables
- future CLI ask LLM wiring
- future API ask LLM wiring
- future ask workflow answer generation

DOWNSTREAM:
- OpenAI LLM adapter
- LLM client boundary
- grounded answer generation service
- future generated ask responses

OWNS:
- provider name normalization
- configured LLM client selection
- missing provider configuration errors
- unsupported provider errors
- disabled provider behavior

DOES_NOT_OWN:
- OpenAI SDK request execution
- prompt construction
- grounded answer generation
- stale-index refusal
- retrieval sufficiency decisions
- API routing
- CLI argument parsing

SIDE_EFFECTS:
- may import and instantiate provider SDK clients through provider adapters

STATE:
  reads:
    - AppConfig records
    - process environment variables when build_configured_llm_client is called without an explicit environment mapping
  writes:
    - none

NOTES:
- Keep provider selection centralized so CLI and API do not duplicate OpenAI-specific wiring.
- This factory does not call the provider network by itself.
- Generated answers should still only run after stale-index and sufficiency checks pass.
"""

from collections.abc import Mapping

from code_context.config import AppConfig, load_app_config
from code_context.llm import LlmClient, build_disabled_llm_client
from code_context.openai_client import build_openai_llm_client


DISABLED_LLM_PROVIDERS = {"", "none", "disabled", "off"}


class LlmConfigurationError(ValueError):
    """Base error for invalid LLM provider configuration."""


class MissingLlmConfigurationError(LlmConfigurationError):
    """Raised when a selected LLM provider is missing required configuration."""


class UnsupportedLlmProviderError(LlmConfigurationError):
    """Raised when the configured LLM provider is not supported."""


def build_configured_llm_client(environment: Mapping[str, str] | None = None) -> LlmClient:
    config = load_app_config(environment)
    return build_llm_client_from_config(config)


def build_llm_client_from_config(config: AppConfig) -> LlmClient:
    provider = normalize_llm_provider(config.llm_provider)

    if provider is None:
        return build_disabled_llm_client()

    if provider == "openai":
        if not config.openai_api_key:
            raise MissingLlmConfigurationError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
        return build_openai_llm_client(api_key=config.openai_api_key)

    raise UnsupportedLlmProviderError(f"Unsupported LLM provider: {config.llm_provider}")


def normalize_llm_provider(provider: str | None) -> str | None:
    normalized = (provider or "").strip().lower()

    if normalized in DISABLED_LLM_PROVIDERS:
        return None

    return normalized
