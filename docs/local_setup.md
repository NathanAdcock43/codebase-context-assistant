# Local Setup Guide

This guide explains how to run AI Codebase Context Assistant locally from a cloned repository.

The project is local-first. It does not require cloud deployment, authentication, a frontend, or a public service. The current version demonstrates grounded codebase indexing, local retrieval, stale-index detection, FastAPI endpoints, CLI commands, an optional LangGraph adapter, and optional generated answers after grounding checks pass.

## Prerequisites

Install these first:

- Python 3.13
- Git
- PowerShell on Windows, or a standard shell on macOS/Linux
- A local clone of this repository

Recommended but optional:

- OpenAI API key for generated answers
- GitKraken or another Git client
- VS Code, PyCharm, or IntelliJ
- Postman or FastAPI Swagger UI for API exploration

## Clone the repository

From the directory where you keep local source code:

    git clone git@github.com:NathanAdcock43/codebase-context-assistant.git
    cd codebase-context-assistant

HTTPS clone also works:

    git clone https://github.com/NathanAdcock43/codebase-context-assistant.git
    cd codebase-context-assistant

## Create a virtual environment on Windows

Create a local virtual environment:

    py -3.13 -m venv .venv

Activate it:

    .\.venv\Scripts\Activate.ps1

Upgrade pip:

    .\.venv\Scripts\python.exe -m pip install --upgrade pip

Install the project in editable mode:

    .\.venv\Scripts\python.exe -m pip install -e .

Install with optional OpenAI support if you want generated answers:

    .\.venv\Scripts\python.exe -m pip install -e ".[openai]"

## Create a virtual environment on macOS or Linux

Create a local virtual environment:

    python3.13 -m venv .venv

Activate it:

    source .venv/bin/activate

Upgrade pip:

    python -m pip install --upgrade pip

Install the project in editable mode:

    python -m pip install -e .

Install with optional OpenAI support if you want generated answers:

    python -m pip install -e ".[openai]"

## Optional OpenAI environment setup

Generated answers are optional. Deterministic indexing, retrieval, stale-index detection, and deterministic ask commands work without an API key.

To enable generated answers, copy the example environment file.

Windows:

    Copy-Item .env.example .env

macOS or Linux:

    cp .env.example .env

Then set `OPENAI_API_KEY` in `.env`.

You may also set `OPENAI_MODEL` in `.env` as the configured default model. CLI and API ask requests can still override that model for one request with `--llm-model` or `llm_model`.

On macOS or Linux, load `.env` into the current shell before running CLI or API generated-answer commands:

    set -a
    source .env
    set +a

On Windows PowerShell, set environment variables for the current session or use your preferred local `.env` loading approach. Do not commit `.env`.

## Validate the project

Run the test suite.

Windows:

    py -3.13 -m pytest

macOS or Linux:

    python -m pytest

Expected result:

    all tests pass

Regenerate the system map.

Windows:

    py -3.13 .\src\code_context\scripts\build_system_map.py

macOS or Linux:

    python src/code_context/scripts/build_system_map.py

Expected result shape:

    Scanned ... source files.
    Wrote docs/system_map.md
    Wrote docs/system_map.json

The scanned file count may increase as the project grows.

## CLI workflow

Set paths for the current repository.

Windows:

    $RepoPath = (Get-Location).Path
    $IndexDir = Join-Path $RepoPath ".code_context_index"

macOS or Linux:

    RepoPath="$(pwd)"
    IndexDir=".code_context_index"

Remove an old local index when starting fresh.

Windows:

    Remove-Item -Recurse -Force $IndexDir -ErrorAction SilentlyContinue

macOS or Linux:

    rm -rf "$IndexDir"

Index the repository.

Windows:

    py -3.13 -m code_context.cli index --repo $RepoPath --index-dir $IndexDir

macOS or Linux:

    python -m code_context.cli index --repo "$RepoPath" --index-dir "$IndexDir"

Ask a grounded deterministic question.

Windows:

    py -3.13 -m code_context.cli ask --index-dir $IndexDir --question "Where is the FastAPI app created?" --no-langgraph

macOS or Linux:

    python -m code_context.cli ask --index-dir "$IndexDir" --question "Where is the FastAPI app created?" --no-langgraph

Ask with an optional generated answer after loading OpenAI environment settings.

Windows:

    py -3.13 -m code_context.cli ask --index-dir $IndexDir --question "Where is the CLI generated answer opt-in implemented?" --limit 8 --no-langgraph --use-llm

macOS or Linux:

    python -m code_context.cli ask --index-dir "$IndexDir" --question "Where is the CLI generated answer opt-in implemented?" --limit 8 --no-langgraph --use-llm

Ask with a per-request model override.

Windows:

    py -3.13 -m code_context.cli ask --index-dir $IndexDir --question "Where is the CLI generated answer opt-in implemented?" --limit 8 --no-langgraph --use-llm --llm-model gpt-5.5

macOS or Linux:

    python -m code_context.cli ask --index-dir "$IndexDir" --question "Where is the CLI generated answer opt-in implemented?" --limit 8 --no-langgraph --use-llm --llm-model gpt-5.5

Check drift.

Windows:

    py -3.13 -m code_context.cli drift --index-dir $IndexDir

macOS or Linux:

    python -m code_context.cli drift --index-dir "$IndexDir"

Show unchanged files in the drift report.

Windows:

    py -3.13 -m code_context.cli drift --index-dir $IndexDir --show-unchanged

macOS or Linux:

    python -m code_context.cli drift --index-dir "$IndexDir" --show-unchanged

## API workflow

Start the local API.

Windows:

    py -3.13 -m uvicorn code_context.api:app --reload

macOS or Linux:

    python -m uvicorn code_context.api:app --reload

Open FastAPI docs:

    http://127.0.0.1:8000/docs

Useful endpoints:

    GET /health
    POST /index
    POST /search
    POST /retrieve
    POST /ask
    POST /drift

Generated-answer API requests require OpenAI environment settings to be loaded before starting Uvicorn.

Example `/ask` request body:

    {
      "index_dir": ".code_context_index",
      "question": "Where is the CLI generated answer opt-in implemented?",
      "limit": 8,
      "use_llm": true
    }

Example `/ask` request body with a per-request model override:

    {
      "index_dir": ".code_context_index",
      "question": "Where is the CLI generated answer opt-in implemented?",
      "limit": 8,
      "use_llm": true,
      "llm_model": "gpt-5.5"
    }

Expected response indicators:

    {
      "confidence": "grounded_generated",
      "is_grounded": true,
      "is_stale": false,
      "is_llm_generated": true,
      "llm_provider": "openai"
    }

## What currently works

The current local project can:

- scan supported source files
- collect file metadata
- calculate file hashes
- chunk source files with line references
- persist a JSON index
- retrieve relevant source chunks
- detect stale indexed context
- refuse when indexed context is stale
- refuse when retrieved context is insufficient
- expose the workflow through FastAPI
- run index, ask, and drift from the CLI
- route ask workflow execution through an optional LangGraph adapter with deterministic fallback
- optionally generate answers through OpenAI after stale-index and grounding checks pass
- downgrade generated self-refusals when the supplied context is insufficient

## Current limitations

- Generated answers are optional and require OpenAI configuration.
- The deterministic answer text is intentionally basic.
- The current vector search is deterministic local retrieval, not ChromaDB yet.
- The project is local-only and single-user.
- The current chunking is line-based rather than AST-aware.
- The system map is useful for visibility, but it is not yet a full C4 architecture export.
- There is no Docker packaging yet.
- There is no authentication or multi-user support.

## Suggested first demo after setup

Run this sequence.

Windows:

    py -3.13 -m pytest
    py -3.13 .\src\code_context\scripts\build_system_map.py
    py -3.13 -m code_context.cli index --repo $RepoPath --index-dir $IndexDir
    py -3.13 -m code_context.cli ask --index-dir $IndexDir --question "Where is the ask workflow implemented?" --no-langgraph
    py -3.13 -m code_context.cli drift --index-dir $IndexDir

macOS or Linux:

    python -m pytest
    python src/code_context/scripts/build_system_map.py
    python -m code_context.cli index --repo "$RepoPath" --index-dir "$IndexDir"
    python -m code_context.cli ask --index-dir "$IndexDir" --question "Where is the ask workflow implemented?" --no-langgraph
    python -m code_context.cli drift --index-dir "$IndexDir"

For the full walkthrough, use:

    docs/demo_script.md

## Troubleshooting

If Python cannot import `code_context`, confirm you are running commands from the repository root and that the project was installed in editable mode.

If PowerShell blocks virtual environment activation, run:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

If ask returns stale-index refusal, re-index the repository.

Windows:

    py -3.13 -m code_context.cli index --repo $RepoPath --index-dir $IndexDir

macOS or Linux:

    python -m code_context.cli index --repo "$RepoPath" --index-dir "$IndexDir"

If retrieval returns insufficient context, ask a narrower question or verify that the repository was indexed.

If generated answers fail, confirm:

- the OpenAI optional dependency is installed
- `OPENAI_API_KEY` is set in the current shell or API process
- `LLM_PROVIDER=openai`
- the requested model is available to your OpenAI account
- the deterministic version of the same ask request works first

If FastAPI docs do not load, confirm `uvicorn` is installed and that the API command is running from the repository root.

## Development workflow

For project changes, use the normal slice workflow:

1. create or replace files
2. run tests
3. regenerate the system map
4. review `git status`
5. stage changes in GitKraken
6. commit with the slice label
7. push

Standard validation commands.

Windows:

    py -3.13 -m pytest
    py -3.13 .\src\code_context\scripts\build_system_map.py
    git status

macOS or Linux:

    python -m pytest
    python src/code_context/scripts/build_system_map.py
    git status
