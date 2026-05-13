# Project Brief

AI Codebase Context Assistant is a small local developer tool for understanding a codebase with grounded, inspectable context.

The project is built around one reliability question:

    How can AI help a developer understand a codebase without drifting away from the actual code?

## What it does

The tool scans a local repository, collects file metadata, calculates file hashes, chunks source files with line references, stores a local JSON index, retrieves relevant code context, and answers developer questions with file references.

It also checks whether the index has gone stale. If files have changed since indexing, the ask workflow refuses to answer until the repository is re-indexed.

Generated answers are optional. The default behavior is deterministic. When generated answers are enabled, the system still checks for stale indexed context, retrieves grounded source chunks, verifies whether the context is sufficient, and only then calls the configured LLM provider.

## What it demonstrates

This project demonstrates a practical AI workflow for codebase understanding:

- local repository scanning
- file metadata and hash tracking
- line-preserving source chunking
- local JSON index persistence
- deterministic retrieval over indexed chunks
- retrieval sufficiency checks
- stale-index detection
- refusal behavior when context is stale or insufficient
- FastAPI endpoints for indexing, retrieval, ask, and drift checks
- CLI commands for local indexing, asking, and drift checks
- optional LangGraph routing around planner, retriever, verifier, and responder steps
- optional OpenAI-generated answers after grounding checks pass
- generated-answer self-refusal downgrades when the supplied context is insufficient

## Why it is intentionally small

This is not a production AI platform. It is a focused engineering project that makes the reliability boundary visible.

The important design choice is that LLM generation is not the foundation. The foundation is source scanning, indexing, retrieval, verification, citation, stale-context detection, and refusal behavior. LLM output is added only after those guardrails are in place.

That makes the project easier to test, explain, and extend.

## Current architecture

The main workflow is:

    repository files
        -> scanner
        -> chunker
        -> JSON index store
        -> local retrieval
        -> planner
        -> retriever
        -> verifier
        -> responder
        -> deterministic answer or refusal
        -> optional generated answer after verification

The stale-context workflow is:

    indexed file metadata
        -> current file scan
        -> hash comparison
        -> drift report
        -> stale-index refusal when needed

## Demo-ready capabilities

A local demo can show:

1. running the test suite
2. regenerating the system map
3. indexing the repository
4. asking deterministic grounded questions
5. showing cited source files and line ranges
6. changing a file to create drift
7. detecting stale indexed context
8. refusing to answer while stale
9. re-indexing
10. asking again successfully
11. optionally calling OpenAI for a generated answer after grounding checks pass
12. using FastAPI `/ask` with `use_llm: true`

## Good demo questions

- Where is the FastAPI app created?
- Where is the ask workflow implemented?
- How does the CLI index command work?
- How does the system detect stale indexed context?
- What files would I change to adjust source chunking?
- What files would I change to adjust retrieval scoring?
- Where is the optional LangGraph adapter implemented?
- Where is the CLI generated answer opt-in implemented?

## Current limitations

- The project is local-only and single-user.
- Generated answers require optional OpenAI configuration.
- The current vector search is a deterministic local baseline, not ChromaDB yet.
- The current chunking is line-based rather than AST-aware.
- There is no authentication or multi-user support.
- There is no Docker packaging yet.
- The system map is useful for visibility, but it is not yet a full C4 architecture export.
- Generated-answer citation verification can be improved further.

## Future work

Good next improvements include:

- add ChromaDB-backed vector storage
- add a small sample repository or fixture for demos
- improve generated-answer citation verification
- add AST-aware chunking
- add incremental re-indexing
- add Git-aware change detection
- add evidence-backed architecture model export
- add Docker packaging after local setup is stable
- add GitHub Actions validation
