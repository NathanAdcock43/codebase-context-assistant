from pathlib import Path

from pydantic import BaseModel, Field


class FileMetadata(BaseModel):
    """Metadata collected for a source file during repository scanning."""

    path: Path
    relative_path: str
    extension: str
    size_bytes: int
    modified_at: float
    content_hash: str
    language: str = Field(default="unknown")
