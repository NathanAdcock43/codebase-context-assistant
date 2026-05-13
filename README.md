# AI Codebase Context Assistant

AI Codebase Context Assistant is a small local developer tool for understanding a codebase with grounded, inspectable context.

The project scans a repository, stores file metadata and line-preserving source chunks, retrieves relevant code context, and answers developer questions with file references. It is built around one core reliability question:

    How can AI help a developer understand a codebase without drifting away from the actual code?

The current version is intentionally local-first and explainable. It does not call an LLM yet. It proves the foundation first: indexing, retrieval, grounding, refusal behavior, stale-index detection, FastAPI endpoints, CLI commands, and an optional LangGraph adapter.

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
- PowerShell

Optional:

- GitKraken or another Git client
- Postman or FastAPI Swagger UI
- PyCharm, IntelliJ, VS Code, or another editor

## Quick start

Clone the repository:

    git clone git@github.com:NathanAdcock43/codebase-context-assistant.git
    cd codebase-context-assistant

Create a virtual environment:

    py -3.13 -m venv .venv

Activate it:

    .\.venv\Scripts\Activate.ps1

Install locally:

    .\.venv\Scripts\python.exe -m pip install --upgrade pip
    .\.venv\Scripts\python.exe -m pip install -e .

Run tests:

    py -3.13 -m pytest

Regenerate the system map:

    py -3.13 .\src\code_context\scripts\build_system_map.py

For detailed setup instructions, see:

    docs\local_setup.md

## CLI usage

Set local paths:

    $RepoPath = (Get-Location).Path
    $IndexDir = Join-Path $RepoPath ".code_context_index"

Create an index:

    py -3.13 -m code_context.cli index --repo $RepoPath --index-dir $IndexDir

Ask a grounded question:

    py -3.13 -m code_context.cli ask --index-dir $IndexDir --question "Where is the ask workflow implemented?" --no-langgraph

Check whether the index is stale:

    py -3.13 -m code_context.cli drift --index-dir $IndexDir

Show unchanged files too:

    py -3.13 -m code_context.cli drift --index-dir $IndexDir --show-unchanged

## FastAPI usage

Start the local API:

    py -3.13 -m uvicorn code_context.api:app --reload

Open the API docs:

    http://127.0.0.1:8000/docs

Current endpoints:

- `GET /health`
- `POST /index`
- `POST /search`
- `POST /retrieve`
- `POST /ask`
- `POST /drift`

## Demo

Use the local demo script:

    docs\demo_script.md

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

## Example questions

Good demo questions:

- Where is the FastAPI app created?
- Where is the ask workflow implemented?
- How does the CLI index command work?
- How does the system detect stale indexed context?
- What files would I change to adjust source chunking?
- What files would I change to adjust retrieval scoring?
- Where is the optional LangGraph adapter implemented?

## Current limitations

- The project does not call an LLM yet.
- The current answer text is deterministic and basic.
- The current vector search is a local deterministic baseline, not ChromaDB yet.
- The project is local-only and single-user.
- The current chunking is line-based rather than AST-aware.
- There is no Docker packaging yet.
- There is no authentication or multi-user support.
- The system map is useful for project visibility, but it is not yet a full C4 architecture export.

## Next likely improvements

Near-term improvements:

- add configurable LLM answer generation after verification
- preserve stale-index refusal before any LLM call
- preserve citations and source references in generated answers
- add ChromaDB-backed vector storage
- improve README and demo polish after LLM integration
- add `.env.example` for LLM provider settings
- add a small sample repository or fixture for demos

Later improvements:

- AST-aware chunking
- incremental re-indexing
- Git-aware change detection
- evidence-backed architecture model export
- C4-friendly architecture export
- Docker packaging
- GitHub Actions validation

## Development workflow

For project changes:

1. create or replace files using `Write-Utf8NoBom`
2. run tests
3. regenerate the system map
4. review `git status`
5. stage changes in GitKraken
6. commit with the slice label
7. push

Standard validation commands:

    py -3.13 -m pytest
    py -3.13 .\src\code_context\scripts\build_system_map.py
    git status

## Project positioning

This is a practical AI workflow tooling project, not a production AI platform.

It demonstrates a grounded local workflow for codebase understanding: scan, index, retrieve, verify, answer, and refuse when the evidence is stale or insufficient.