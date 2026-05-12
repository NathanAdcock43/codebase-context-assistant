from __future__ import annotations

"""
ROLE: Provide local command-line entry points for codebase context workflows.
LAYER: cli
FLOW: local_cli

INPUTS:
- command-line arguments
- local JSON index directory
- developer question text
- retrieval limit
- retrieval score thresholds
- retrieval sufficiency thresholds

OUTPUTS:
- plain-text ask results for terminal users
- process exit codes for command success or operational failure

UPSTREAM:
- local developer terminal usage
- future demo scripts
- future README examples

DOWNSTREAM:
- JSON index store
- reusable ask workflow service
- optional LangGraph workflow adapter through ask service
- deterministic agent workflow fallback through ask service

OWNS:
- CLI argument parsing
- CLI command dispatch
- terminal output formatting
- CLI error reporting
- process exit code behavior

DOES_NOT_OWN:
- repository indexing pipeline
- JSON persistence internals
- stale-index detection
- retrieval scoring
- ask workflow orchestration
- optional LangGraph workflow behavior
- deterministic agent workflow behavior
- LLM prompting

SIDE_EFFECTS:
- reads a local JSON index file
- writes terminal output
- writes terminal error output

STATE:
  reads:
    - local JSON index file
  writes:
    - terminal stdout
    - terminal stderr

NOTES:
- Keep CLI behavior thin and reuse application services.
- Stale-index and insufficient-context refusals are successful command executions because the tool behaved correctly.
- Operational failures such as missing index files should return a non-zero exit code.
"""

import argparse
import sys
from collections.abc import Sequence

from pydantic import ValidationError

from code_context.ask import AskWorkflowResult, ask_indexed_code_question
from code_context.index_store import DEFAULT_INDEX_FILENAME, JsonIndexStore


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ask":
        return _run_ask_command(args)

    parser.print_help()
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="code-context",
        description="Local CLI for AI Codebase Context Assistant workflows.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a question against an existing local code context index.",
    )
    ask_parser.add_argument(
        "--index-dir",
        required=True,
        help="Directory containing the local JSON index.",
    )
    ask_parser.add_argument(
        "--question",
        required=True,
        help="Developer question to answer from indexed context.",
    )
    ask_parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Maximum number of retrieved chunks to use.",
    )
    ask_parser.add_argument(
        "--min-score",
        type=float,
        default=0.15,
        help="Minimum retrieval score for candidate chunks.",
    )
    ask_parser.add_argument(
        "--minimum-results",
        type=int,
        default=1,
        help="Minimum number of relevant chunks required to answer.",
    )
    ask_parser.add_argument(
        "--minimum-top-score",
        type=float,
        default=0.2,
        help="Minimum top result score required to answer.",
    )
    ask_parser.add_argument(
        "--index-filename",
        default=DEFAULT_INDEX_FILENAME,
        help="Index filename inside the index directory.",
    )
    ask_parser.add_argument(
        "--no-langgraph",
        action="store_true",
        help="Use the deterministic workflow directly instead of preferring LangGraph.",
    )

    return parser


def _run_ask_command(args: argparse.Namespace) -> int:
    store = JsonIndexStore(args.index_dir, index_filename=args.index_filename)

    try:
        snapshot = store.load()
        result = ask_indexed_code_question(
            question=args.question,
            snapshot=snapshot,
            limit=args.limit,
            min_score=args.min_score,
            minimum_results=args.minimum_results,
            minimum_top_score=args.minimum_top_score,
            prefer_langgraph=not args.no_langgraph,
        )
    except (FileNotFoundError, NotADirectoryError, ValidationError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(format_ask_result(result), end="")
    return 0


def format_ask_result(result: AskWorkflowResult) -> str:
    """Format an ask result for terminal output."""
    lines = [
        f"Question: {result.question}",
        f"Confidence: {result.confidence}",
        f"Grounded: {_yes_no(result.is_grounded)}",
        f"Stale: {_yes_no(result.is_stale)}",
        "",
        "Answer:",
        result.answer,
    ]

    if result.insufficient_reason:
        lines.extend(
            [
                "",
                "Reason:",
                result.insufficient_reason,
            ]
        )

    if result.plan:
        lines.append("")
        lines.append("Plan:")
        lines.extend(f"{position}. {step}" for position, step in enumerate(result.plan, start=1))

    if result.citations:
        lines.append("")
        lines.append("Citations:")
        lines.extend(f"- {citation}" for citation in result.citations)

    if result.sources:
        lines.append("")
        lines.append("Sources:")
        lines.extend(
            (
                f"- {source.chunk.relative_path}:{source.chunk.start_line}-{source.chunk.end_line} "
                f"(score={source.score:.3f})"
            )
            for source in result.sources
        )

    if result.steps:
        lines.append("")
        lines.append("Workflow steps:")
        lines.extend(
            f"- {step.name}: {step.status.value}"
            for step in result.steps
        )

    return "\n".join(lines).rstrip() + "\n"


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    raise SystemExit(main())