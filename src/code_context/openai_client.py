from __future__ import annotations

"""
ROLE: Adapt the OpenAI Responses API to the provider-neutral LLM client boundary.
LAYER: llm
FLOW: openai_llm_adapter

INPUTS:
- provider-neutral LlmRequest records
- optional OpenAI SDK client instances
- optional OpenAI API keys
- OpenAI Responses API response objects
- optional temperature settings from provider-neutral requests

OUTPUTS:
- provider-neutral LlmResponse records
- OpenAiLlmClient instances
- clear unavailable-client errors when the OpenAI SDK is not installed
- clear response-shape errors when provider output is missing text
- OpenAI requests that omit unsupported sampling parameters for selected models

UPSTREAM:
- LLM client boundary
- AppConfig provider settings
- grounded answer generation service
- API ask LLM integration
- CLI ask LLM integration
- local tests with fake OpenAI clients

DOWNSTREAM:
- generated ask responses
- FastAPI ask LLM integration
- CLI ask LLM integration
- provider configuration documentation

OWNS:
- OpenAI Responses API adapter behavior
- OpenAI client construction helper
- provider-neutral request to OpenAI request conversion
- OpenAI response text extraction
- OpenAI usage metadata extraction
- OpenAI response model fallback behavior
- OpenAI-specific optional parameter filtering

DOES_NOT_OWN:
- prompt construction
- answer generation orchestration
- retrieval scoring
- retrieval sufficiency decisions
- stale-index refusal
- API routing
- CLI argument parsing
- LangGraph workflow routing
- environment variable parsing

SIDE_EFFECTS:
- build_openai_llm_client may import the OpenAI SDK
- OpenAiLlmClient.complete calls the injected OpenAI client

STATE:
  reads:
    - OpenAI SDK availability when build_openai_llm_client is called without a client factory
    - OpenAI environment configuration through the SDK when no API key is passed
  writes:
    - none

NOTES:
- Tests use fake clients and do not make network calls.
- Keep this adapter behind the LlmClient protocol so provider details do not leak into prompt or answer generation code.
- GPT-5 and o-series models may reject temperature, so this adapter omits temperature for those model families.
"""

from collections.abc import Callable, Mapping
from typing import Any

from code_context.llm import LlmClient, LlmRequest, LlmResponse


class OpenAiClientUnavailableError(RuntimeError):
    """Raised when the OpenAI SDK cannot be loaded."""


class OpenAiResponseError(RuntimeError):
    """Raised when an OpenAI response cannot be converted into LlmResponse."""


class OpenAiLlmClient:
    """Provider-neutral adapter around an OpenAI SDK client."""

    def __init__(self, client: Any, *, provider: str = "openai") -> None:
        self._client = client
        self._provider = provider

    def complete(self, request: LlmRequest) -> LlmResponse:
        request_kwargs: dict[str, Any] = {
            "model": request.model,
            "input": _to_openai_input(request),
        }

        if _should_send_temperature(request):
            request_kwargs["temperature"] = request.temperature

        response = self._client.responses.create(**request_kwargs)

        return LlmResponse(
            content=_extract_output_text(response),
            model=_extract_response_model(response, fallback_model=request.model),
            provider=self._provider,
            usage=_extract_usage(response),
        )


def build_openai_llm_client(
    *,
    api_key: str | None = None,
    client_factory: Callable[..., Any] | None = None,
) -> LlmClient:
    """Build an OpenAI-backed LlmClient.

    A client_factory can be supplied by tests to avoid importing the SDK or
    making network calls.
    """

    factory = client_factory or _load_openai_client_factory()

    if api_key is None:
        client = factory()
    else:
        client = factory(api_key=api_key)

    return OpenAiLlmClient(client)


def _load_openai_client_factory() -> Callable[..., Any]:
    try:
        from openai import OpenAI
    except ImportError as exc:
        msg = "OpenAI SDK is not installed. Install it with: pip install openai"
        raise OpenAiClientUnavailableError(msg) from exc

    return OpenAI


def _to_openai_input(request: LlmRequest) -> list[dict[str, str]]:
    return [{"role": message.role, "content": message.content} for message in request.messages]


def _should_send_temperature(request: LlmRequest) -> bool:
    return request.temperature is not None and _model_supports_temperature(request.model)


def _model_supports_temperature(model: str) -> bool:
    normalized_model = model.strip().lower()

    unsupported_prefixes = (
        "gpt-5",
        "o1",
        "o3",
        "o4",
    )

    return not normalized_model.startswith(unsupported_prefixes)


def _extract_output_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)

    if not isinstance(output_text, str) or not output_text.strip():
        msg = "OpenAI response did not include output_text."
        raise OpenAiResponseError(msg)

    return output_text.strip()


def _extract_response_model(response: Any, *, fallback_model: str) -> str:
    model = getattr(response, "model", None)

    if isinstance(model, str) and model.strip():
        return model.strip()

    return fallback_model


def _extract_usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage", None)

    if usage is None:
        return {}

    if isinstance(usage, Mapping):
        return {str(key): value for key, value in usage.items() if isinstance(value, int)}

    metadata: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens", "prompt_tokens", "completion_tokens"):
        value = getattr(usage, key, None)
        if isinstance(value, int):
            metadata[key] = value

    return metadata
