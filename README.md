# AI Codebase Context Assistant

AI Codebase Context Assistant is a small local developer tool for understanding a codebase with grounded, inspectable context.

The project scans a repository, stores file metadata and line-preserving source chunks, retrieves relevant code context, and answers developer questions with file references. It is built around one core reliability question:

    How can AI help a developer understand a codebase without drifting away from the actual code?

The current version is intentionally local-first and explainable. It defaults to deterministic answers, and generated answers are opt-in through the CLI or API after stale-index and grounding checks pass. It proves the foundation first: indexing, retrieval, grounding, refusal behavior, stale-index detection, FastAPI endpoints, CLI commands, optional generated answers, and an optional LangGraph adapter.

## Current capabilities

The project can currently:

- scan supported source files
- skip ignored directories
- calculate file hashes
- collect file metadata
- chunk source files with start and end line references
- persist a local JSON index
- retrieve relevant source chunks
- decide whether retrieved context is sufficient
- refuse when context is insufficient
- detect stale indexed context after files change
- refuse to answer from a stale index
- expose workflows through FastAPI
- run local index, ask, and drift commands through the CLI
- route ask workflow execution through an optional LangGraph adapter with deterministic fallback
- optionally call a configured OpenAI LLM after stale-index and grounding checks pass
- downgrade generated self-refusals when the supplied context is insufficient

## Current architecture

The current workflow is:

    repository files
        -> scanner
        -> chunker
        -> JSON index store
        -> local retrieval
        -> planner
        -> retriever
        -> verifier
        -> responder
        -> grounded answer or refusal
        -> optional generated answer after verification

The stale-context workflow is:

    indexed file metadata
        -> current file scan
        -> hash comparison
        -> drift report
        -> stale-index refusal when needed

Key modules:

- `src/code_context/scanner.py`: scans repository files and creates file metadata
- `src/code_context/chunker.py`: chunks source files while preserving line references
- `src/code_context/index_store.py`: saves and loads JSON index snapshots
- `src/code_context/drift.py`: compares indexed metadata against current files
- `src/code_context/vector_store.py`: provides deterministic local retrieval over chunks
- `src/code_context/retrieval.py`: retrieves grounded context and checks sufficiency
- `src/code_context/ask.py`: orchestrates ask-time stale checks and workflow execution
- `src/code_context/agent/`: contains planner, retriever, verifier, responder workflow pieces
- `src/code_context/api.py`: exposes FastAPI endpoints
- `src/code_context/cli.py`: exposes local CLI commands
- `src/code_context/scripts/build_system_map.py`: generates project architecture map outputs

## Requirements

- Python 3.13
- Git
- PowerShell on Windows, or a standard shell on macOS/Linux

Optional:

- OpenAI API key and OpenAI Python SDK for generated answers
- GitKraken or another Git client
- Postman or FastAPI Swagger UI
- PyCharm, IntelliJ, VS Code, or another editor

## Quick start

Clone the repository:

    git clone git@github.com:NathanAdcock43/codebase-context-assistant.git
    cd codebase-context-assistant

Create and activate a virtual environment.

Windows:

    py -3.13 -m venv .venv
    .\.venv\Scripts\Activate.ps1

macOS or Linux:

    python3.13 -m venv .venv
    source .venv/bin/activate

Install locally.

Windows:

    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -e .

macOS or Linux:

    python -m pip install --upgrade pip
    python -m pip install -e .

Install with optional OpenAI support if you want generated answers.

Windows:

    .\.venv\Scripts\python.exe -m pip install -e ".[openai]"

macOS or Linux:

    python -m pip install -e ".[openai]"

Run tests:

    python -m pytest

Regenerate the system map:

    python src/code_context/scripts/build_system_map.py

For detailed setup instructions, see:

    docs/local_setup.md

## CLI usage

Set local paths.

Windows:

    $RepoPath = (Get-Location).Path
    $IndexDir = Join-Path $RepoPath ".code_context_index"

macOS or Linux:

    RepoPath="$(pwd)"
    IndexDir=".code_context_index"

Create an index:

    python -m code_context.cli index --repo "$RepoPath" --index-dir "$IndexDir"

Ask a grounded question:

    python -m code_context.cli ask --index-dir "$IndexDir" --question "Where is the ask workflow implemented?" --no-langgraph

Ask with an optional generated answer after loading OpenAI environment settings:

    python -m code_context.cli ask --index-dir "$IndexDir" --question "Where is the CLI generated answer opt-in implemented?" --limit 8 --no-langgraph --use-llm --llm-temperature 0

Check whether the index is stale:

    python -m code_context.cli drift --index-dir "$IndexDir"

Show unchanged files too:

    python -m code_context.cli drift --index-dir "$IndexDir" --show-unchanged

## FastAPI usage

Start the local API:

    python -m uvicorn code_context.api:app --reload

Open the API docs:

    http://127.0.0.1:8000/docs

Current endpoints:

- `GET /health`
- `POST /index`
- `POST /search`
- `POST /retrieve`
- `POST /ask`
- `POST /drift`

Optional generated-answer `/ask` request body:

    {
      "index_dir": ".code_context_index",
      "question": "Where is the CLI generated answer opt-in implemented?",
      "limit": 8,
      "use_llm": true,
      "llm_temperature": 0
    }

Generated answers still use the same stale-index and grounding checks before the provider is called.

## Demo

Use the local demo script:

    docs/demo_script.md

Use the release-ready checklist:

    docs/demo_checklist.md

The demo walks through:

1. running tests
2. refreshing the system map
3. indexing the repository
4. asking grounded questions
5. modifying a file
6. detecting stale context
7. refusing to answer from a stale index
8. re-indexing
9. asking again
10. optionally asking through generated CLI and API paths

## Example questions

Good demo questions:

- Where is the FastAPI app created?
- Where is the ask workflow implemented?
- How does the CLI index command work?
- How does the system detect stale indexed context?
- What files would I change to adjust source chunking?
- What files would I change to adjust retrieval scoring?
- Where is the optional LangGraph adapter implemented?
- Where is the CLI generated answer opt-in implemented?

## Current limitations

- Generated answers are optional and require OpenAI configuration.
- Deterministic answer text is intentionally basic.
- The current vector search is a local deterministic baseline, not ChromaDB yet.
- The project is local-only and single-user.
- The current chunking is line-based rather than AST-aware.
- There is no Docker packaging yet.
- There is no authentication or multi-user support.
- The system map is useful for project visibility, but it is not yet a full C4 architecture export.
- Generated-answer citation verification can be improved further.

## Next likely improvements

Near-term improvements:

- add ChromaDB-backed vector storage
- add a small sample repository or fixture for demos
- improve generated-answer citation verification
- improve API demo examples
- add Docker packaging when local setup is stable

Later improvements:

- AST-aware chunking
- incremental re-indexing
- Git-aware change detection
- evidence-backed architecture model export
- C4-friendly architecture export
- GitHub Actions validation

## Development workflow

For project changes:

1. create or replace files
2. run tests
3. regenerate the system map
4. review `git status`
5. stage changes in GitKraken
6. commit with the slice label
7. push

Standard validation commands:

    python -m pytest
    python src/code_context/scripts/build_system_map.py
    git status

## Project positioning

This is a practical AI workflow tooling project, not a production AI platform.

It demonstrates a grounded local workflow for codebase understanding: scan, index, retrieve, verify, answer, and refuse when the evidence is stale or insufficient.
