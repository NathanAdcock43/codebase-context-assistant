# AI Codebase Context Assistant Demo Script

This demo shows the current local workflow for the AI Codebase Context Assistant.

The project is still intentionally small and local. The current workflow does not call an LLM yet. It demonstrates the foundation that makes later LLM answers safer:

1. scan a local repository
2. chunk source files with line references
3. persist an index
4. retrieve relevant context
5. answer only from grounded context
6. refuse when context is stale or insufficient
7. show a planner, retriever, verifier, responder workflow

## Demo setup

Run these commands from the project root:

    cd C:\TCC\LocalSources\NathansUtils\codebase-context-assistant

Use the current repository as the demo target:

    $RepoPath = "C:\TCC\LocalSources\NathansUtils\codebase-context-assistant"
    $IndexDir = ".\.code_context_index"

Optional cleanup before a fresh demo:

    Remove-Item -Recurse -Force $IndexDir -ErrorAction SilentlyContinue

## Step 1: Run tests

    py -3.13 -m pytest

Expected result:

    all tests pass

This proves the scanner, chunker, index store, drift detector, retrieval service, API, agent workflow, LangGraph adapter, and CLI are currently working.

## Step 2: Build or refresh the system map

    py -3.13 .\src\code_context\scripts\build_system_map.py

Expected result:

    Scanned 33 source files.
    Wrote docs\system_map.md
    Wrote docs\system_map.json

The system map is part of the project discipline. It helps keep the codebase explainable as slices are added.

## Step 3: Index the repository from the CLI

    py -3.13 -m code_context.cli index --repo $RepoPath --index-dir $IndexDir

Expected output shape:

    Indexed repository:
    C:\TCC\LocalSources\NathansUtils\codebase-context-assistant

    Index path: .code_context_index\code_context_index.json
    Indexed at: ...
    File count: ...
    Chunk count: ...

What to say:

    The CLI calls the same repository indexing pipeline used by the API. It scans supported files, calculates hashes, chunks the source with line references, and writes a local JSON index.

## Step 4: Ask a grounded question

    py -3.13 -m code_context.cli ask --index-dir $IndexDir --question "Where is the FastAPI app created?" --no-langgraph

Expected output shape:

    Question: Where is the FastAPI app created?
    Confidence: grounded
    Grounded: yes
    Stale: no

    Answer:
    ...

    Citations:
    - ...

    Sources:
    - src/code_context/api.py:...

What to say:

    The assistant is not free-form guessing. It retrieves indexed source chunks, verifies whether the retrieved context is sufficient, and returns file and line references.

## Step 5: Ask a workflow question

    py -3.13 -m code_context.cli ask --index-dir $IndexDir --question "How does the ask workflow detect stale context?" --no-langgraph

Expected behavior:

    The answer should refer to the ask service, scanner, and drift detector when retrieval finds enough context.

What to say:

    This is the central reliability feature. Before answering, the ask workflow compares the indexed snapshot against the current repository state. If the files have changed, it refuses instead of answering from stale context.

## Step 6: Modify a file to create drift

Use a harmless temporary change. For example, add a comment to a Python file, then save it.

Example:

    Add-Content -Path ".\src\code_context\config.py" -Value "# temporary demo drift"

Do not commit this temporary change.

## Step 7: Run drift detection

    py -3.13 -m code_context.cli drift --index-dir $IndexDir

Expected output shape:

    Drift status: stale
    Stale: yes

    Added files: 0
    Modified files: 1
    Removed files: 0

    Modified:
    - src/code_context/config.py

What to say:

    The index knows which file hashes were captured during indexing. After the file changes, drift detection reports that the index is stale.

## Step 8: Try to ask again while stale

    py -3.13 -m code_context.cli ask --index-dir $IndexDir --question "Where is local configuration defined?" --no-langgraph

Expected output shape:

    Confidence: stale_index
    Grounded: no
    Stale: yes

    Answer:
    I cannot answer from this index because the indexed context is stale. Re-index the repository and ask again.

    Reason:
    Indexed context is stale. Re-index the repository before asking questions.

What to say:

    This is the key point of the project. The assistant refuses when indexed context has drifted from the actual code.

## Step 9: Revert the temporary change

If you used the example comment above:

    git restore .\src\code_context\config.py

Then confirm the working tree state:

    git status

## Step 10: Re-index and ask again

    py -3.13 -m code_context.cli index --repo $RepoPath --index-dir $IndexDir

Then ask again:

    py -3.13 -m code_context.cli ask --index-dir $IndexDir --question "Where is local configuration defined?" --no-langgraph

Expected result:

    The stale refusal should be gone after the index matches the current repository again.

## Optional API demo

Start the API locally:

    py -3.13 -m uvicorn code_context.api:app --reload

Then use FastAPI docs:

    http://127.0.0.1:8000/docs

Useful endpoints:

    GET /health
    POST /index
    POST /search
    POST /retrieve
    POST /ask
    POST /drift

## Demo talk track

Use this simple explanation:

    This is a local developer tool that helps answer questions about a codebase without drifting away from the actual files.

    The system scans the repository, stores file hashes and line-preserving chunks, retrieves relevant code context, verifies whether the retrieved context is enough, and refuses when it cannot answer safely.

    The current version uses a deterministic local retrieval baseline and an optional LangGraph adapter around the planner, retriever, verifier, and responder workflow. The next natural step is to add LLM answer generation after the verifier, while preserving the same grounding and stale-index refusal behavior.

## Current honest limitations

- The current answer text is deterministic and basic.
- The project does not call an LLM yet.
- The current vector search is a local deterministic baseline, not ChromaDB yet.
- The project is local-only and single-user.
- The current chunking is line-based rather than AST-aware.
- The system map is useful for project visibility, but it is not yet a full C4 architecture export.

## Good demo questions

- Where is the FastAPI app created?
- Where is the ask workflow implemented?
- How does the CLI index command work?
- How does the system detect stale indexed context?
- What files would I change to adjust source chunking?
- What files would I change to adjust retrieval scoring?
- Where is the optional LangGraph adapter implemented?