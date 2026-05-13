# Demo Checklist

This checklist records the known-good local demo paths for AI Codebase Context Assistant.

Use it when preparing to show or explain the project.

## Baseline validation

Run from the project root.

Windows:

    py -3.13 -m pytest
    py -3.13 .\src\code_context\scripts\build_system_map.py

macOS or Linux:

    python -m pytest
    python src/code_context/scripts/build_system_map.py

Expected result:

    all tests pass
    system map is regenerated successfully

## Prepare local paths

Windows:

    $RepoPath = (Get-Location).Path
    $IndexDir = Join-Path $RepoPath ".code_context_index"

macOS or Linux:

    RepoPath="$(pwd)"
    IndexDir=".code_context_index"

## Index the repository

Windows:

    py -3.13 -m code_context.cli index --repo $RepoPath --index-dir $IndexDir

macOS or Linux:

    python -m code_context.cli index --repo "$RepoPath" --index-dir "$IndexDir"

Expected result:

    Indexed repository:
    <repo path>

    Index path: <index path>
    File count: ...
    Chunk count: ...

## Deterministic CLI ask

Windows:

    py -3.13 -m code_context.cli ask --index-dir $IndexDir --question "Where is the ask workflow implemented?" --no-langgraph

macOS or Linux:

    python -m code_context.cli ask --index-dir "$IndexDir" --question "Where is the ask workflow implemented?" --no-langgraph

Expected indicators:

    Confidence: grounded
    Grounded: yes
    Stale: no
    Sources include src/code_context/ask.py

## Generated CLI ask

This requires optional OpenAI setup.

macOS or Linux environment load:

    set -a
    source .env
    set +a

Windows users should set equivalent environment variables in the active PowerShell session.

Windows:

    py -3.13 -m code_context.cli ask --index-dir $IndexDir --question "Where is the CLI generated answer opt-in implemented?" --limit 8 --no-langgraph --use-llm --llm-temperature 0

macOS or Linux:

    python -m code_context.cli ask --index-dir "$IndexDir" --question "Where is the CLI generated answer opt-in implemented?" --limit 8 --no-langgraph --use-llm --llm-temperature 0

Expected indicators:

    Confidence: grounded_generated
    Grounded: yes
    Stale: no
    LLM generated: yes
    LLM provider: openai
    Sources include src/code_context/cli.py

## Deterministic API ask

Start the API.

Windows:

    py -3.13 -m uvicorn code_context.api:app --reload

macOS or Linux:

    python -m uvicorn code_context.api:app --reload

In a second terminal, call `/ask`:

    curl -s --max-time 20 -X POST "http://127.0.0.1:8000/ask" \
      -H "Content-Type: application/json" \
      -d '{
        "index_dir": ".code_context_index",
        "question": "Where is the CLI generated answer opt-in implemented?",
        "limit": 8,
        "use_llm": false
      }' | python -m json.tool

Expected indicators:

    "confidence": "grounded"
    "is_grounded": true
    "is_stale": false
    "is_llm_generated": false
    sources include src/code_context/cli.py

## Generated API ask

Generated API requests require OpenAI environment settings to be loaded before starting Uvicorn.

    curl -sS --max-time 90 \
      -o /tmp/code_context_api_ask_llm.json \
      -w "HTTP status: %{http_code}\nTotal time: %{time_total}s\n" \
      -X POST "http://127.0.0.1:8000/ask" \
      -H "Content-Type: application/json" \
      -d '{
        "index_dir": ".code_context_index",
        "question": "Where is the CLI generated answer opt-in implemented?",
        "limit": 8,
        "use_llm": true,
        "llm_temperature": 0
      }'

    python -m json.tool /tmp/code_context_api_ask_llm.json

Expected indicators:

    HTTP status: 200
    "confidence": "grounded_generated"
    "is_grounded": true
    "is_stale": false
    "is_llm_generated": true
    "llm_provider": "openai"
    sources include src/code_context/cli.py

A known-good local smoke run returned in about 4.2 seconds with model `gpt-4.1-mini-2025-04-14`.

## Stale-index refusal

Create a temporary change.

Windows:

    Add-Content -Path ".\src\code_context\config.py" -Value "# temporary demo drift"

macOS or Linux:

    printf '\n# temporary demo drift\n' >> src/code_context/config.py

Check drift.

Windows:

    py -3.13 -m code_context.cli drift --index-dir $IndexDir

macOS or Linux:

    python -m code_context.cli drift --index-dir "$IndexDir"

Expected indicators:

    Drift status: stale
    Stale: yes
    Modified includes src/code_context/config.py

Ask while stale.

Windows:

    py -3.13 -m code_context.cli ask --index-dir $IndexDir --question "Where is local configuration defined?" --no-langgraph

macOS or Linux:

    python -m code_context.cli ask --index-dir "$IndexDir" --question "Where is local configuration defined?" --no-langgraph

Expected indicators:

    Confidence: stale_index
    Grounded: no
    Stale: yes
    Re-index the repository before asking questions.

Revert the temporary change.

Windows:

    git restore .\src\code_context\config.py

macOS or Linux:

    git restore src/code_context/config.py

Then re-index before continuing.

## Demo summary

The project is ready to explain when these points are true:

- tests pass
- system map regenerates
- deterministic CLI ask works
- generated CLI ask works when OpenAI is configured
- deterministic API ask works
- generated API ask works when OpenAI is configured
- stale-index refusal works
- README, setup guide, project brief, and demo script match the current behavior
