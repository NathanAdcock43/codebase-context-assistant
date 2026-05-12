# Architecture

## Planned components

- Scanner: walks a repository and collects source file metadata.
- Chunker: splits source files while preserving line references.
- Index store: persists file metadata and indexed chunk records.
- Drift detector: compares current files against indexed hashes and timestamps.
- Vector store: stores and retrieves searchable source chunks.
- API: exposes local FastAPI endpoints.
- Agent workflow: planner, retriever, verifier, and responder.

## MVP principle

Build the reliable codebase context foundation first. Add AI workflow logic only after scanning, hashing, chunking, persistence, and drift detection are working.
