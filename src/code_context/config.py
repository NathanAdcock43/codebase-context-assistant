from __future__ import annotations

"""
ROLE: Define local configuration values for the codebase context assistant.
LAYER: config
FLOW: configuration

INPUTS:
- local environment defaults
- process environment variables
- project index directory setting
- index filename setting
- default text encoding setting
- LLM provider settings
- future embedding provider settings
- local API settings

OUTPUTS:
- AppConfig records
- load_app_config helper results

UPSTREAM:
- local development setup
- .env.example documentation
- future CLI commands
- future FastAPI startup configuration
- generated answer workflow configuration

DOWNSTREAM:
- scanner
- index store
- vector store
- API configuration
- CLI configuration
- LLM provider integration
- local tooling

OWNS:
- local application configuration model
- default index directory
- default index filename
- default file encoding
- environment variable parsing
- typed runtime configuration shape
- provider configuration defaults

DOES_NOT_OWN:
- repository scanning
- file hashing
- chunking
- vector search
- LLM provider clients
- embedding provider clients
- FastAPI route definitions
- CLI argument parsing

SIDE_EFFECTS:
- load_app_config reads process environment when no explicit environment mapping is provided

STATE:
  reads:
    - process environment variables when load_app_config is called without an explicit environment mapping
  writes:
    - none

NOTES:
- Keep this model small and explicit.
- Do not load .env files directly in this module.
- CLI and API ask requests can override the configured model per request.
- Keep the source default stable; use OPENAI_MODEL, CLI flags, or API fields for model experiments.
"""

import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


DEFAULT_OPENAI_MODEL = "gpt-4.1-mini"


class AppConfig(BaseModel):
    """Typed local runtime configuration."""

    environment: str = "local"
    index_dir: Path = Field(default=Path(".code_context_index"))
    index_filename: str = "code_context_index.json"
    default_encoding: str = "utf-8"

    llm_provider: str = "openai"
    openai_api_key: str | None = None
    openai_model: str = DEFAULT_OPENAI_MODEL

    embedding_provider: str = "local"
    embedding_model: str | None = None
    chroma_persist_dir: Path = Field(default=Path(".chroma"))

    api_host: str = "127.0.0.1"
    api_port: int = 8000

    @field_validator("environment", "index_filename", "default_encoding", "llm_provider", "openai_model", "embedding_provider", "api_host")
    @classmethod
    def _non_empty_string(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            msg = "Configuration value cannot be empty."
            raise ValueError(msg)
        return normalized

    @field_validator("api_port")
    @classmethod
    def _valid_api_port(cls, value: int) -> int:
        if value < 1 or value > 65535:
            msg = "API port must be between 1 and 65535."
            raise ValueError(msg)
        return value


def load_app_config(environment: Mapping[str, str] | None = None) -> AppConfig:
    """Load app configuration from an environment mapping.

    Passing an explicit mapping keeps tests deterministic. If no mapping is
    provided, this reads from process environment variables.
    """

    env = os.environ if environment is None else environment

    return AppConfig(
        environment=_get_env(env, "CODE_CONTEXT_ENV", "local"),
        index_dir=Path(_get_env(env, "CODE_CONTEXT_INDEX_DIR", ".code_context_index")),
        index_filename=_get_env(env, "CODE_CONTEXT_INDEX_FILENAME", "code_context_index.json"),
        default_encoding=_get_env(env, "CODE_CONTEXT_DEFAULT_ENCODING", "utf-8"),
        llm_provider=_get_env(env, "LLM_PROVIDER", "openai"),
        openai_api_key=_get_optional_env(env, "OPENAI_API_KEY"),
        openai_model=_get_env(env, "OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        embedding_provider=_get_env(env, "EMBEDDING_PROVIDER", "local"),
        embedding_model=_get_optional_env(env, "EMBEDDING_MODEL"),
        chroma_persist_dir=Path(_get_env(env, "CHROMA_PERSIST_DIR", ".chroma")),
        api_host=_get_env(env, "API_HOST", "127.0.0.1"),
        api_port=_get_int_env(env, "API_PORT", 8000),
    )


def _get_env(environment: Mapping[str, str], key: str, default: str) -> str:
    value = environment.get(key)
    if value is None or not value.strip():
        return default
    return value.strip()


def _get_optional_env(environment: Mapping[str, str], key: str) -> str | None:
    value = environment.get(key)
    if value is None or not value.strip():
        return None
    return value.strip()


def _get_int_env(environment: Mapping[str, str], key: str, default: int) -> int:
    value = environment.get(key)
    if value is None or not value.strip():
        return default

    try:
        return int(value.strip())
    except ValueError as exc:
        msg = f"{key} must be an integer."
        raise ValueError(msg) from exc