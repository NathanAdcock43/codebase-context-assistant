# Architecture Notes

AI Codebase Context Assistant is a local developer tool for understanding a codebase through grounded context retrieval.

The project is built around this reliability question:

    How can AI help a developer understand a codebase without drifting away from the actual code?

The current architecture proves the non-LLM foundation first:

1. scan local source files
2. collect metadata and content hashes
3. chunk files while preserving line references
4. persist an index snapshot
5. retrieve relevant code chunks
6. verify whether retrieved context is sufficient
7. refuse when context is insufficient
8. detect when indexed context is stale
9. refuse to answer from stale indexed context

The project currently exposes this workflow through both FastAPI and a local CLI.

## Current architecture status

The current version includes:

- repository scanner
- line-preserving chunker
- JSON index store
- drift detector
- deterministic local vector-style retrieval
- retrieval sufficiency checks
- deterministic planner, retriever, verifier, responder workflow
- optional LangGraph workflow adapter
- reusable ask orchestration service
- FastAPI endpoints
- CLI commands for index, ask, and drift
- system map generation script
- local setup and demo documentation

The current version does not call an LLM yet.

That is intentional. The project protects grounding, stale-index refusal, and deterministic test coverage before adding generated answer text.

## High-level workflow

The normal indexing flow is:

    local repository
        -> scanner
        -> file metadata with hashes
        -> chunker
        -> source chunks with line references
        -> JSON index snapshot

The normal ask flow is:

    developer question
        -> load index snapshot
        -> scan current repository files
        -> detect drift
        -> refuse if index is stale
        -> planner
        -> retriever
        -> verifier
        -> responder
        -> grounded answer or refusal

The retrieval flow is:

    question
        -> deterministic local vector-style search
        -> ranked source chunks
        -> sufficiency check
        -> retrieval response

The stale-index flow is:

    indexed file metadata
        -> current file metadata
        -> compare relative paths and hashes
        -> added, modified, removed, unchanged file groups
        -> stale or current report

## Core modules

### `src/code_context/scanner.py`

Scans a local repository and returns file metadata for supported source files.

Owns:

- repository traversal
- ignored directory filtering
- supported extension filtering
- file hash calculation
- basic language inference
- `FileMetadata` construction

Does not own:

- source chunking
- index persistence
- drift comparison
- retrieval
- API routing
- AI behavior

### `src/code_context/chunker.py`

Splits source files into deterministic chunks while preserving file and line references.

Owns:

- line-based chunking
- start and end line references
- overlap handling
- stable chunk ID generation

Does not own:

- repository scanning
- AST parsing
- metadata persistence
- retrieval
- answer generation

### `src/code_context/index_store.py`

Persists and loads JSON index snapshots.

Owns:

- index directory creation
- JSON snapshot writing
- JSON snapshot loading
- `IndexSnapshot` serialization and validation

Does not own:

- repository traversal
- file hashing
- chunking
- drift comparison
- retrieval

### `src/code_context/drift.py`

Compares indexed file metadata against current file metadata.

Owns:

- added file detection
- removed file detection
- modified file detection by content hash
- unchanged file reporting
- stale index determination

Does not own:

- scanning files directly
- hashing files directly
- API routing
- re-indexing decisions

### `src/code_context/vector_store.py`

Provides deterministic local vector-style retrieval.

Owns:

- tokenization
- local hashed embedding generation
- cosine similarity
- ranked chunk retrieval
- score threshold filtering

Does not own:

- ChromaDB persistence
- provider-backed embeddings
- LLM prompting
- answer generation

This is a local retrieval baseline. It gives the project reliable tests before adding ChromaDB.

### `src/code_context/retrieval.py`

Retrieves grounded source context and decides whether it is sufficient.

Owns:

- retrieval orchestration over indexed chunks
- sufficiency checks
- insufficient-context reasons
- preserving grounded search results

Does not own:

- answer generation
- drift detection
- API routing
- vector scoring internals

### `src/code_context/agent/state.py`

Defines explicit state models for the code question workflow.

Owns:

- `CodeQuestionState`
- `AgentStep`
- `VerificationResult`
- initial state construction
- step status updates

Does not own:

- planning logic
- retrieval logic
- response logic
- LangGraph edge definitions

### `src/code_context/agent/nodes.py`

Defines deterministic planner, retriever, verifier, and responder nodes.

Owns:

- default plan creation
- retrieval node orchestration
- verification decision shaping
- grounded citation construction
- refusal response construction

Does not own:

- retrieval scoring internals
- generated natural-language synthesis
- API routing
- LangGraph graph construction

### `src/code_context/agent/graph.py`

Runs the deterministic workflow in a fixed order.

Current order:

    planner
        -> retriever
        -> verifier
        -> responder

Owns:

- deterministic workflow sequencing
- final state return

Does not own:

- node internals
- LangGraph wiring
- API routing

### `src/code_context/agent/langgraph_graph.py`

Provides an optional LangGraph adapter around the already-tested deterministic nodes.

Owns:

- LangGraph availability detection
- LangGraph workflow construction
- deterministic fallback selection
- LangGraph node wrapping

Does not own:

- node internals
- API response shaping
- LLM prompting

The adapter keeps LangGraph integration low-risk because the deterministic workflow remains usable when LangGraph is not installed.

### `src/code_context/ask.py`

Orchestrates grounded question answering over an indexed repository snapshot.

Owns:

- ask-time stale-index detection
- stale-index refusal construction
- optional LangGraph adapter invocation
- confidence classification for ask results
- reusable result shape for API and CLI callers

Does not own:

- HTTP DTOs
- API routing
- scanner internals
- drift comparison internals
- agent node behavior
- LLM prompting

This is the application-service layer for asking questions.

### `src/code_context/api.py`

Exposes the workflow through FastAPI.

Current endpoints:

- `GET /health`
- `POST /index`
- `POST /search`
- `POST /retrieve`
- `POST /ask`
- `POST /drift`

Owns:

- FastAPI app creation
- request and response DTOs
- route definitions
- HTTP error handling
- API response shaping

Does not own:

- scanner internals
- index persistence internals
- retrieval sufficiency logic
- ask workflow orchestration
- drift comparison logic

### `src/code_context/cli.py`

Exposes local terminal commands.

Current commands:

- `index`
- `ask`
- `drift`

Owns:

- CLI argument parsing
- command dispatch
- terminal output formatting
- CLI error reporting
- process exit code behavior

Does not own:

- repository traversal
- index persistence internals
- stale-index comparison logic
- ask workflow orchestration
- retrieval scoring

## Data models

The shared models live in `src/code_context/models.py`.

Current models include:

- `FileMetadata`
- `SourceChunk`
- `IndexSnapshot`
- `FileDrift`
- `DriftReport`
- `SearchResult`
- `RetrievalResponse`

These models are intentionally simple. They are the contract between scanner, chunker, persistence, drift detection, retrieval, API, CLI, and agent workflow modules.

## Grounding behavior

Grounding currently means:

- retrieved answers must come from indexed source chunks
- source chunks preserve file path, start line, and end line
- answers include citations or source references
- insufficient retrieval produces a refusal
- stale indexed context produces a refusal before workflow execution

The system should not answer from stale or weak context.

## Stale-index behavior

Ask-time stale detection is one of the key reliability features.

Before answering, the ask service:

1. loads the stored index snapshot
2. scans the current repository
3. compares current file metadata against indexed metadata
4. refuses if files were added, modified, or removed

This prevents the assistant from answering from an index that no longer matches the actual repository.

## CLI and API entry points

The CLI and API share core services.

The CLI is useful for demos and local usage:

    py -3.13 -m code_context.cli index --repo $RepoPath --index-dir $IndexDir
    py -3.13 -m code_context.cli ask --index-dir $IndexDir --question "Where is the ask workflow implemented?" --no-langgraph
    py -3.13 -m code_context.cli drift --index-dir $IndexDir

The API is useful for local HTTP workflows and FastAPI docs:

    py -3.13 -m uvicorn code_context.api:app --reload

Then open:

    http://127.0.0.1:8000/docs

## System map discipline

The system map script lives at:

    src/code_context/scripts/build_system_map.py

It scans Python files, extracts architecture headers, imports, classes, and functions, then writes:

    docs/system_map.md
    docs/system_map.json

This keeps the codebase inspectable as slices are added.

The system map is not yet a full C4 export. It is a local project visibility tool and a foundation for future architecture export work.

## Current limitations

The current architecture has clear limits:

- no LLM call yet
- deterministic answer text is basic
- local vector-style retrieval is not ChromaDB yet
- line-based chunking is not AST-aware
- JSON persistence is simple and local
- no authentication
- no multi-user support
- no Docker packaging yet
- no cloud deployment
- no production security claims
- system map output is not yet a complete evidence-backed C4 architecture model

## Future architecture direction

Near-term additions should preserve the current reliability behavior.

Likely next steps:

1. add configurable LLM answer generation after verification
2. preserve stale-index refusal before any LLM call
3. preserve citations and source references in generated answers
4. add ChromaDB-backed vector storage
5. add `.env`-backed runtime configuration
6. improve demo polish after LLM integration

Later additions:

- AST-aware chunking
- incremental re-indexing
- Git-aware change detection
- Git event triggered re-indexing
- evidence-backed architecture model export
- C4-friendly JSON export
- Structurizr, C4-PlantUML, Mermaid, or IcePanel-style export
- Docker packaging
- GitHub Actions validation

## Design principles

Keep the project:

- local-first
- explainable
- testable
- grounded in actual files
- honest about limitations
- easy to run from GitHub
- small enough to explain clearly

Do not add complexity before the foundation supports it.