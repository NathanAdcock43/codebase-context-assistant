from pathlib import Path

from pydantic import BaseModel, Field


class AppConfig(BaseModel):
    """Local application configuration."""

    index_dir: Path = Field(default=Path(".code_context_index"))
    default_encoding: str = Field(default="utf-8")
