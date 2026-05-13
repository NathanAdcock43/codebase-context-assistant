from __future__ import annotations

"""
ROLE: Define provider-neutral LLM request, response, and client boundary models.
LAYER: llm
FLOW: llm_client_boundary

INPUTS:
- future grounded answer prompts
- future provider configuration
- model names
- message lists
- temperature settings

OUTPUTS:
- LlmMessage records
- LlmRequest records
- LlmResponse records
- LlmClient protocol
- DisabledLlmClient placeholder behavior

UPSTREAM:
- future answer generation service
- future responder node
- future OpenAI provider adapter
- AppConfig provider settings
- local tests

DOWNSTREAM:
- future LLM answer generation
- future grounded response synthesis
- future provider-specific clients
- future API and CLI ask responses

OWNS:
- provider-neutral LLM message shape
- provider-neutral LLM request shape
- provider-neutral LLM response shape
- LLM client protocol
- disabled client placeholder behavior
- shared LLM validation rules

DOES_NOT_OWN:
- OpenAI SDK calls
- prompt construction
- retrieval
- grounding verification
- stale-index refusal
- API routing
- CLI argument parsing
- LangGraph workflow routing

SIDE_EFFECTS:
- DisabledLlmClient raises an error when called

STATE:
  reads:
    - none
  writes:
    - none

NOTES:
- This module does not call an LLM yet.
- Keep this provider-neutral so OpenAI or another provider can be added behind the same boundary.
- Future answer generation must still run only after stale-index and sufficiency checks pass.
"""

from collections.abc import Mapping
from typing import Literal, Protocol

from pydantic import BaseModel, Field, field_validator


LlmRole = Literal["system", "user", "assistant"]


class LlmClientUnavailableError(RuntimeError):
    """Raised when LLM generation is requested before a provider client exists."""


class LlmMessage(BaseModel):
    """Provider-neutral chat message."""

    role: LlmRole
    content: str

    @field_validator("content")
    @classmethod
    def _content_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            msg = "LLM message content cannot be empty."
            raise ValueError(msg)
        return normalized


class LlmRequest(BaseModel):
    """Provider-neutral LLM completion request."""

    messages: list[LlmMessage]
    model: str
    temperature: float = 0.2

    @field_validator("messages")
    @classmethod
    def _messages_must_not_be_empty(cls, value: list[LlmMessage]) -> list[LlmMessage]:
        if not value:
            msg = "LLM request must include at least one message."
            raise ValueError(msg)
        return value

    @field_validator("model")
    @classmethod
    def _model_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            msg = "LLM model cannot be empty."
            raise ValueError(msg)
        return normalized

    @field_validator("temperature")
    @classmethod
    def _temperature_must_be_in_range(cls, value: float) -> float:
        if value < 0 or value > 2:
            msg = "LLM temperature must be between 0 and 2."
            raise ValueError(msg)
        return value


class LlmResponse(BaseModel):
    """Provider-neutral LLM completion response."""

    content: str
    model: str
    provider: str
    usage: dict[str, int] = Field(default_factory=dict)

    @field_validator("content", "model", "provider")
    @classmethod
    def _required_text_must_not_be_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            msg = "LLM response text fields cannot be empty."
            raise ValueError(msg)
        return normalized


class LlmClient(Protocol):
    """Protocol implemented by future provider-specific LLM clients."""

    def complete(self, request: LlmRequest) -> LlmResponse:
        """Return a completion for the given request."""


class DisabledLlmClient:
    """Placeholder client used until real LLM generation is wired in."""

    def complete(self, request: LlmRequest) -> LlmResponse:
        raise LlmClientUnavailableError(
            "LLM answer generation is not configured yet. The current workflow should keep using deterministic grounded responses."
        )


def build_disabled_llm_client() -> LlmClient:
    """Return the current placeholder LLM client."""

    return DisabledLlmClient()


def build_usage_metadata(usage: Mapping[str, int] | None = None) -> dict[str, int]:
    """Normalize provider usage metadata into a plain dictionary."""

    if usage is None:
        return {}

    return {key: value for key, value in usage.items()}