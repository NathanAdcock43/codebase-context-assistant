from __future__ import annotations

"""
ROLE: Verify repository scanner behavior for supported files, ignored paths, hashes, and metadata.
LAYER: tests
FLOW: scanner_validation

INPUTS:
- temporary repository directories
- sample source files
- ignored directory examples
- unsupported file examples

OUTPUTS:
- scanner behavior assertions

UPSTREAM:
- scanner implementation
- FileMetadata model

DOWNSTREAM:
- local test runs
- future CI guardrails
- system map review

OWNS:
- scanner unit test coverage
- repository traversal expectations
- hash stability checks
- relative path behavior checks
- language inference checks

DOES_NOT_OWN:
- chunking tests
- drift detection tests
- vector store tests
- API tests
- agent workflow tests

SIDE_EFFECTS:
- writes temporary test files through pytest tmp_path

STATE:
  reads:
    - temporary test files
  writes:
    - temporary test files

NOTES:
- These tests keep the scanner deterministic before adding indexing or AI behavior.
- Use byte writes for test files where size or hash expectations matter.
"""

import hashlib
from pathlib import Path

import pytest

from code_context.scanner import calculate_file_hash, infer_language, scan_repository


def test_scan_repository_returns_supported_files_with_metadata(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    source_dir = repo / "src"
    source_dir.mkdir(parents=True)

    python_content = b"print('hello')\n"
    readme_content = b"# Example\n"

    python_file = source_dir / "app.py"
    python_file.write_bytes(python_content)

    markdown_file = repo / "README.md"
    markdown_file.write_bytes(readme_content)

    results = scan_repository(repo)

    by_relative_path = {item.relative_path: item for item in results}

    assert set(by_relative_path) == {"README.md", "src/app.py"}

    app_metadata = by_relative_path["src/app.py"]
    assert app_metadata.path == python_file.resolve()
    assert app_metadata.extension == ".py"
    assert app_metadata.size_bytes == len(python_content)
    assert app_metadata.content_hash == hashlib.sha256(python_content).hexdigest()
    assert app_metadata.language == "python"

    readme_metadata = by_relative_path["README.md"]
    assert readme_metadata.extension == ".md"
    assert readme_metadata.size_bytes == len(readme_content)
    assert readme_metadata.content_hash == hashlib.sha256(readme_content).hexdigest()
    assert readme_metadata.language == "markdown"


def test_scan_repository_skips_ignored_directories(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    included_file = repo / "main.py"
    included_file.write_bytes(b"print('included')\n")

    ignored_dir = repo / ".venv"
    ignored_dir.mkdir()
    ignored_file = ignored_dir / "ignored.py"
    ignored_file.write_bytes(b"print('ignored')\n")

    node_modules_dir = repo / "node_modules"
    node_modules_dir.mkdir()
    node_module_file = node_modules_dir / "package.js"
    node_module_file.write_bytes(b"console.log('ignored');\n")

    results = scan_repository(repo)

    relative_paths = {item.relative_path for item in results}

    assert relative_paths == {"main.py"}


def test_scan_repository_skips_unsupported_extensions(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    supported_file = repo / "query.sql"
    supported_file.write_bytes(b"select 1;\n")

    unsupported_file = repo / "image.png"
    unsupported_file.write_bytes(b"not really an image")

    results = scan_repository(repo)

    relative_paths = {item.relative_path for item in results}

    assert relative_paths == {"query.sql"}


def test_scan_repository_allows_custom_supported_extensions(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()

    python_file = repo / "main.py"
    python_file.write_bytes(b"print('ignored by custom extension list')\n")

    custom_file = repo / "notes.custom"
    custom_file.write_bytes(b"custom content\n")

    results = scan_repository(repo, supported_extensions={".custom"})

    assert [item.relative_path for item in results] == ["notes.custom"]
    assert results[0].extension == ".custom"
    assert results[0].language == "unknown"


def test_calculate_file_hash_is_stable(tmp_path: Path) -> None:
    content = b"value = 123\n"
    file_path = tmp_path / "sample.py"
    file_path.write_bytes(content)

    first_hash = calculate_file_hash(file_path)
    second_hash = calculate_file_hash(file_path)

    assert first_hash == second_hash
    assert first_hash == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize(
    ("filename", "expected_language"),
    [
        ("app.py", "python"),
        ("README.md", "markdown"),
        ("config.toml", "toml"),
        ("package.json", "json"),
        ("script.js", "javascript"),
        ("component.ts", "typescript"),
        ("Report.java", "java"),
        ("query.sql", "sql"),
        ("unknown.custom", "unknown"),
    ],
)
def test_infer_language(filename: str, expected_language: str) -> None:
    assert infer_language(Path(filename)) == expected_language