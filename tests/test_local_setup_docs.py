from __future__ import annotations

"""
ROLE: Verify local setup documentation for GitHub-oriented project packaging.
LAYER: tests
FLOW: local_setup_documentation_validation

INPUTS:
- docs/local_setup.md content
- expected local setup commands
- expected CLI workflow commands
- expected API workflow commands
- expected limitation language

OUTPUTS:
- documentation content assertions

UPSTREAM:
- local setup documentation
- CLI index command
- CLI ask command
- CLI drift command
- FastAPI app
- demo script

DOWNSTREAM:
- local test runs
- future CI guardrails
- README packaging polish
- GitHub project presentation
- system map review

OWNS:
- local setup documentation smoke tests
- expected setup command coverage
- expected CLI command coverage
- expected API command coverage
- honest limitation wording coverage

DOES_NOT_OWN:
- CLI behavior tests
- API behavior tests
- demo script tests
- scanner tests
- retrieval tests
- drift detector tests
- LangGraph adapter tests

SIDE_EFFECTS:
- reads docs/local_setup.md

STATE:
  reads:
    - docs/local_setup.md
  writes:
    - none

NOTES:
- These tests are intentionally light.
- They protect the local setup guide from losing the minimum commands needed by a new reader.
"""

from pathlib import Path


LOCAL_SETUP = Path("docs/local_setup.md")


def test_local_setup_documents_environment_and_validation_commands() -> None:
    content = LOCAL_SETUP.read_text(encoding="utf-8")

    assert "Python 3.13" in content
    assert "py -3.13 -m venv .venv" in content
    assert ".\\.venv\\Scripts\\python.exe -m pip install -e ." in content
    assert "py -3.13 -m pytest" in content
    assert "py -3.13 .\\src\\code_context\\scripts\\build_system_map.py" in content


def test_local_setup_documents_cli_and_api_workflows() -> None:
    content = LOCAL_SETUP.read_text(encoding="utf-8")

    assert "py -3.13 -m code_context.cli index" in content
    assert "py -3.13 -m code_context.cli ask" in content
    assert "py -3.13 -m code_context.cli drift" in content
    assert "py -3.13 -m uvicorn code_context.api:app --reload" in content
    assert "http://127.0.0.1:8000/docs" in content


def test_local_setup_keeps_current_limitations_honest() -> None:
    content = LOCAL_SETUP.read_text(encoding="utf-8")

    assert "The current version does not call an LLM yet." in content
    assert "not ChromaDB yet" in content
    assert "local-only and single-user" in content
    assert "There is no Docker packaging yet." in content