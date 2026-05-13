# Local Setup Guide

This guide explains how to run AI Codebase Context Assistant locally from a cloned repository.

The project is currently local-first. It does not require cloud deployment, authentication, a frontend, or a public service. The current version demonstrates grounded codebase indexing, local retrieval, stale-index detection, FastAPI endpoints, CLI commands, and an optional LangGraph adapter.

## Prerequisites

Install these first:

- Python 3.13
- Git
- PowerShell
- A local clone of this repository

Recommended but optional:

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

## Create a virtual environment

Create a local virtual environment:

    py -3.13 -m venv .venv

Activate it:

    .\.venv\Scripts\Activate.ps1

Upgrade pip:

    .\.venv\Scripts\python.exe -m pip install --upgrade pip

Install the project in editable mode:

    .\.venv\Scripts\python.exe -m pip install -e .

If editable install is not available in the local project state, install the dependencies listed in `pyproject.toml`, then continue with the validation commands below.

## Validate the project

Run the test suite:

    py -3.13 -m pytest

Expected result:

    all tests pass

Regenerate the system map:

    py -3.13 .\src\code_context\scripts\build_system_map.py

Expected result shape:

    Scanned 35 source files.
    Wrote docs\system_map.md
    Wrote docs\system_map.json

The scanned file count may increase as the project grows.

## CLI workflow

Set paths for the current repository:

    $RepoPath = "C:\TCC\LocalSources\NathansUtils\codebase-context-assistant"
    $IndexDir = ".\.code_context_index"

Remove an old local index when starting fresh:

    Remove-Item -Recurse -Force $IndexDir -ErrorAction SilentlyContinue

Index the repository:

    py -3.13 -m code_context.cli index --repo $RepoPath --index-dir $IndexDir

Ask a grounded question:

    py -3.13 -m code_context.cli ask --index-dir $IndexDir --question "Where is the FastAPI app created?" --no-langgraph

Check drift:

    py -3.13 -m code_context.cli drift --index-dir $IndexDir

Show unchanged files in the drift report:

    py -3.13 -m code_context.cli drift --index-dir $IndexDir --show-unchanged

## API workflow

Start the local API:

    py -3.13 -m uvicorn code_context.api:app --reload

Open FastAPI docs:

    http://127.0.0.1:8000/docs

Useful endpoints:

    GET /health
    POST /index
    POST /search
    POST /retrieve
    POST /ask
    POST /drift

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

## Current limitations

- The current version does not call an LLM yet.
- The answer text is deterministic and basic.
- The current vector search is deterministic local retrieval, not ChromaDB yet.
- The project is local-only and single-user.
- The current chunking is line-based rather than AST-aware.
- The system map is useful for visibility, but it is not yet a full C4 architecture export.
- There is no Docker packaging yet.
- There is no authentication or multi-user support.

## Suggested first demo after setup

Run this sequence:

    py -3.13 -m pytest
    py -3.13 .\src\code_context\scripts\build_system_map.py
    py -3.13 -m code_context.cli index --repo $RepoPath --index-dir $IndexDir
    py -3.13 -m code_context.cli ask --index-dir $IndexDir --question "Where is the ask workflow implemented?" --no-langgraph
    py -3.13 -m code_context.cli drift --index-dir $IndexDir

For the full walkthrough, use:

    docs\demo_script.md

## Troubleshooting

If Python cannot import `code_context`, confirm you are running commands from the repository root.

If PowerShell blocks virtual environment activation, run:

    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

If ask returns stale-index refusal, re-index the repository:

    py -3.13 -m code_context.cli index --repo $RepoPath --index-dir $IndexDir

If retrieval returns insufficient context, ask a narrower question or verify that the repository was indexed.

If FastAPI docs do not load, confirm `uvicorn` is installed and that the API command is running from the repository root.

## Development workflow

For project changes, use the normal slice workflow:

1. create or replace files with `Write-Utf8NoBom`
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