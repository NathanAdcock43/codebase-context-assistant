# AI Codebase Context Assistant

AI Codebase Context Assistant is a small developer tool for understanding a local codebase with AI-assisted context retrieval.

The project scans source files, records metadata and hashes, preserves line references, stores searchable context, and answers developer questions with grounded file references.

## Core question

How can AI help a developer understand a codebase without drifting away from the actual code?

## MVP build order

1. Repo scanner
2. File metadata and hashing
3. Chunking with line references
4. Metadata persistence
5. Drift detection
6. Vector search
7. FastAPI endpoints
8. LangGraph planner, retriever, verifier, responder flow
9. Documentation and demo polish

## Local development

Create a virtual environment:

py -3.13 -m venv .venv

Activate it:

.\.venv\Scripts\Activate.ps1

Install locally with dev tools:

python -m pip install -e ".[dev]"

Run tests:

python -m pytest

## Status

Initial project setup. The first implementation slice will be the repository scanner.
