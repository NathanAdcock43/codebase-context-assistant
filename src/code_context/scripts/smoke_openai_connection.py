from __future__ import annotations

"""
ROLE: Run a manual OpenAI connection smoke test through the project LLM adapter.
LAYER: diagnostics
FLOW: openai_connection_smoke_test

INPUTS:
- OPENAI_API_KEY environment variable
- OPENAI_MODEL environment variable or AppConfig default model
- OpenAI Python SDK installation
- project OpenAI LLM adapter

OUTPUTS:
- terminal confirmation that the OpenAI SDK can be called
- model name used by the provider response
- short generated response text
- optional usage metadata

UPSTREAM:
- local developer terminal usage
- OpenAI client adapter
- environment-backed local setup
- future generated ask workflow wiring

DOWNSTREAM:
- local connection validation
- future LLM ask integration
- future README setup notes
- future demo script updates

OWNS:
- manual OpenAI connectivity verification
- safe environment-variable checks
- small provider call through OpenAiLlmClient
- terminal smoke-test output

DOES_NOT_OWN:
- ask workflow orchestration
- API routing
- CLI ask behavior
- retrieval
- grounding verification
- stale-index refusal
- prompt building for code questions
- automated provider integration tests

SIDE_EFFECTS:
- calls the OpenAI API when run manually
- may consume a small amount of API credit
- writes terminal output

STATE:
  reads:
    - OPENAI_API_KEY environment variable
    - OPENAI_MODEL environment variable
  writes:
    - terminal stdout
    - terminal stderr on failure

NOTES:
- This script is intentionally manual and is not part of pytest's real provider coverage.
- Do not print the API key.
- Keep the request tiny so the smoke test is cheap.
"""

import sys

from code_context.config import load_app_config
from code_context.llm import LlmMessage, LlmRequest
from code_context.openai_client import (
    OpenAiClientUnavailableError,
    OpenAiResponseError,
    build_openai_llm_client,
)


def main() -> int:
    config = load_app_config()

    if not config.openai_api_key:
        print("OPENAI_API_KEY is not set. Load your .env file or export the key before running this script.", file=sys.stderr)
        return 1

    request = LlmRequest(
        model=config.openai_model,
        temperature=0.0,
        messages=[
            LlmMessage(
                role="system",
                content="You are a connection smoke test. Reply with one short sentence.",
            ),
            LlmMessage(
                role="user",
                content="Confirm that the OpenAI connection works for this local codebase assistant.",
            ),
        ],
    )

    try:
        client = build_openai_llm_client(api_key=config.openai_api_key)
        response = client.complete(request)
    except OpenAiClientUnavailableError as exc:
        print(f"OpenAI SDK unavailable: {exc}", file=sys.stderr)
        return 1
    except OpenAiResponseError as exc:
        print(f"OpenAI response error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"OpenAI smoke test failed: {exc}", file=sys.stderr)
        return 1

    print("OpenAI connection smoke test succeeded.")
    print(f"Provider: {response.provider}")
    print(f"Model: {response.model}")
    print(f"Answer: {response.content}")

    if response.usage:
        print(f"Usage: {response.usage}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
