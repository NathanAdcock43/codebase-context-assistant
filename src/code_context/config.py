from __future__ import annotations

"""
ROLE: Define local configuration values for the codebase context assistant.
LAYER: config
FLOW: configuration

INPUTS:
- local environment defaults
- project index directory setting
- default text encoding setting

OUTPUTS:
- AppConfig

UPSTREAM:
- local development setup
- future CLI commands
- future FastAPI startup configuration

DOWNSTREAM:
- scanner
- index store
- vector store
- API configuration
- local tooling

OWNS:
- local application configuration model
- default index directory
- default file encoding

DOES_NOT_OWN:
- repository scanning
- file hashing
- chunking
- vector search
- LLM provider configuration
- FastAPI route definitions

SIDE_EFFECTS:
- none

STATE:
  reads:
    - none
  writes:
    - none

NOTES:
- Keep this model small until runtime configuration needs become clearer.
- Avoid hiding project behavior behind configuration too early.
"""

from pathlib import Path

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """Local application configuration."""

    index_dir: Path = Field(default=Path(".code_context_index"))
    default_encoding: str = Field(default="utf-8")
