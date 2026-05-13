from __future__ import annotations

"""
ROLE: Verify demo documentation for the local CLI workflow.
LAYER: tests
FLOW: demo_documentation_validation

INPUTS:
- docs/demo_script.md content
- expected CLI command examples
- expected stale-index demo language
- expected honest limitation language

OUTPUTS:
- documentation content assertions

UPSTREAM:
- demo documentation
- CLI index command
- CLI ask command
- CLI drift command
- ask workflow stale-index refusal behavior

DOWNSTREAM:
- local test runs
- future CI guardrails
- README and packaging polish
- interview demo preparation
- system map review

OWNS:
- demo script smoke tests
- expected local CLI command coverage
- stale-index demo coverage
- limitation wording coverage

DOES_NOT_OWN:
- CLI behavior tests
- API behavior tests
- scanner tests
- retrieval tests
- drift detector tests
- LangGraph adapter tests
- LLM response tests

SIDE_EFFECTS:
- reads docs/demo_script.md

STATE:
  reads:
    - docs/demo_script.md
  writes:
    - none

NOTES:
- These tests are intentionally light.
- They protect the demo path from being accidentally removed during documentation edits.
"""

from pathlib import Path


DEMO_SCRIPT = Path("docs/demo_script.md")


def test_demo_script_documents_cli_index_ask_and_drift_commands() -> None:
    content = DEMO_SCRIPT.read_text(encoding="utf-8")

    assert "py -3.13 -m code_context.cli index" in content
    assert "py -3.13 -m code_context.cli ask" in content
    assert "py -3.13 -m code_context.cli drift" in content
    assert "--index-dir $IndexDir" in content


def test_demo_script_documents_stale_index_refusal_flow() -> None:
    content = DEMO_SCRIPT.read_text(encoding="utf-8")

    assert "Confidence: stale_index" in content
    assert "Indexed context is stale. Re-index the repository before asking questions." in content
    assert "refuses when indexed context has drifted from the actual code" in content


def test_demo_script_keeps_limitations_honest() -> None:
    content = DEMO_SCRIPT.read_text(encoding="utf-8")

    assert "The project does not call an LLM yet." in content
    assert "not ChromaDB yet" in content
    assert "local-only and single-user" in content