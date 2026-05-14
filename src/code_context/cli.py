from __future__ import annotations

"""
ROLE: Provide local command-line entry points for codebase context workflows.
LAYER: cli
FLOW: local_cli

INPUTS:
- command-line arguments
- local repository path
- local JSON index directory
- developer question text
- indexing settings
- retrieval limit
- retrieval score thresholds
- retrieval sufficiency thresholds
- optional repository override for drift checks
- optional generated-answer ask flags, including per-request model overrides

OUTPUTS:
- plain-text indexing summaries for terminal users
- plain-text ask results for terminal users
- optional LLM metadata in ask output when generated answers are requested
- plain-text drift reports for terminal users
- process exit codes for command success or operational failure

UPSTREAM:
- local developer terminal usage
- future demo scripts
- future README examples

DOWNSTREAM:
- repository indexing pipeline
- repository scanner
- drift detector
- JSON index store
- reusable ask workflow service
- optional LangGraph workflow adapter through ask service
- deterministic agent workflow fallback through ask service
- configured LLM client factory through ask service
- grounded answer generation through ask service

OWNS:
- CLI argument parsing
- CLI command dispatch
- terminal output formatting
- CLI error reporting
- process exit code behavior
- generated-answer ask flag and model override forwarding
- generated-answer terminal metadata formatting

DOES_NOT_OWN:
- repository traversal rules
- file hashing internals
- source chunking internals
- JSON persistence internals
- stale-index comparison logic
- retrieval scoring
- ask workflow orchestration
- optional LangGraph workflow behavior
- deterministic agent workflow behavior
- prompt formatting
- provider-specific SDK calls

SIDE_EFFECTS:
- index command reads repository files and writes a local JSON index file
- ask command reads a local JSON index file and repository files through the ask service
- ask command may call a configured LLM client when --use-llm is passed and ask service guardrails pass
- drift command reads a local JSON index file and repository files
- writes terminal output
- writes terminal error output

STATE:
  reads:
    - local repository files
    - local JSON index file
    - process environment through ask service when generated answers are requested
  writes:
    - local JSON index file through index command
    - terminal stdout
    - terminal stderr

NOTES:
- Keep CLI behavior thin and reuse application services.
- Stale-index and insufficient-context refusals are successful command executions because the tool behaved correctly.
- Generated answers must remain opt-in through --use-llm.
- CLI callers may pass --llm-model to override the configured model for one request.
- Operational failures such as missing repository paths or missing index files should return a non-zero exit code.
"""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from code_context.ask import AskWorkflowResult, ask_indexed_code_question
from code_context.drift import detect_drift
from code_context.index_store import DEFAULT_INDEX_FILENAME, JsonIndexStore
from code_context.llm import LlmClientUnavailableError
from code_context.models import DriftReport, FileDrift, IndexSnapshot
from code_context.pipeline import index_repository
from code_context.scanner import scan_repository


def main(argv: Sequence[str] | None = None) -> int:
    """Run the local CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "index":
        return _run_index_command(args)

    if args.command == "ask":
        return _run_ask_command(args)

    if args.command == "drift":
        return _run_drift_command(args)

    parser.print_help()
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="code-context",
        description="Local CLI for AI Codebase Context Assistant workflows.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser(
        "index",
        help="Index a local repository into a JSON code context index.",
    )
    index_parser.add_argument(
        "--repo",
        required=True,
        help="Repository directory to scan and index.",
    )
    index_parser.add_argument(
        "--index-dir",
        required=True,
        help="Directory where the local JSON index should be written.",
    )
    index_parser.add_argument(
        "--max-lines",
        type=int,
        default=80,
        help="Maximum source lines per indexed chunk.",
    )
    index_parser.add_argument(
        "--overlap-lines",
        type=int,
        default=10,
        help="Number of overlapping source lines between chunks.",
    )
    index_parser.add_argument(
        "--index-filename",
        default=DEFAULT_INDEX_FILENAME,
        help="Index filename inside the index directory.",
    )

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
    ask_parser.add_argument(
        "--use-llm",
        action="store_true",
        help="Opt in to a generated answer after stale-index and grounding checks pass.",
    )
    ask_parser.add_argument(
        "--llm-model",
        default=None,
        help="Optional LLM model override for generated answers.",
    )
    ask_parser.add_argument(
        "--llm-temperature",
        type=float,
        default=None,
        help="Optional LLM temperature for generated answers. Must be between 0.0 and 2.0 when provided.",
    )

    drift_parser = subparsers.add_parser(
        "drift",
        help="Compare the current repository files against an existing local index.",
    )
    drift_parser.add_argument(
        "--index-dir",
        required=True,
        help="Directory containing the local JSON index.",
    )
    drift_parser.add_argument(
        "--repo",
        default=None,
        help="Optional repository directory override. Defaults to the repo path stored in the index.",
    )
    drift_parser.add_argument(
        "--index-filename",
        default=DEFAULT_INDEX_FILENAME,
        help="Index filename inside the index directory.",
    )
    drift_parser.add_argument(
        "--show-unchanged",
        action="store_true",
        help="Include unchanged files in the terminal drift report.",
    )

    return parser


def _run_index_command(args: argparse.Namespace) -> int:
    try:
        snapshot = index_repository(
            Path(args.repo),
            index_dir=Path(args.index_dir),
            max_lines=args.max_lines,
            overlap_lines=args.overlap_lines,
            index_filename=args.index_filename,
        )
    except (FileNotFoundError, NotADirectoryError, ValidationError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    store = JsonIndexStore(args.index_dir, index_filename=args.index_filename)
    print(format_index_result(snapshot, store.index_path), end="")
    return 0


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
            use_llm=args.use_llm,
            llm_model=args.llm_model,
            llm_temperature=args.llm_temperature,
        )
    except (FileNotFoundError, NotADirectoryError, ValidationError, LlmClientUnavailableError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(format_ask_result(result), end="")
    return 0


def _run_drift_command(args: argparse.Namespace) -> int:
    store = JsonIndexStore(args.index_dir, index_filename=args.index_filename)

    try:
        snapshot = store.load()
        repo_path = Path(args.repo) if args.repo else Path(snapshot.repo_root)
        current_files = scan_repository(repo_path)
        report = detect_drift(snapshot=snapshot, current_files=current_files)
    except (FileNotFoundError, NotADirectoryError, ValidationError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(format_drift_report(report, show_unchanged=args.show_unchanged), end="")
    return 0


def format_index_result(snapshot: IndexSnapshot, index_path: Path) -> str:
    """Format an index result for terminal output."""
    lines = [
        "Indexed repository:",
        snapshot.repo_root,
        "",
        f"Index path: {index_path}",
        f"Indexed at: {snapshot.indexed_at}",
        f"File count: {len(snapshot.files)}",
        f"Chunk count: {len(snapshot.chunks)}",
    ]

    return "\n".join(lines).rstrip() + "\n"


def format_ask_result(result: AskWorkflowResult) -> str:
    """Format an ask result for terminal output."""
    lines = [
        f"Question: {result.question}",
        f"Confidence: {result.confidence}",
        f"Grounded: {_yes_no(result.is_grounded)}",
        f"Stale: {_yes_no(result.is_stale)}",
        f"LLM generated: {_yes_no(result.is_llm_generated)}",
    ]

    if result.is_llm_generated:
        lines.extend(
            [
                f"LLM provider: {result.llm_provider or 'unknown'}",
                f"LLM model: {result.llm_model or 'unknown'}",
            ]
        )

        if result.llm_usage:
            lines.append(f"LLM usage: {result.llm_usage}")

    lines.extend(
        [
            "",
            "Answer:",
            result.answer,
        ]
    )

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


def format_drift_report(report: DriftReport, *, show_unchanged: bool = False) -> str:
    """Format a drift report for terminal output."""
    lines = [
        f"Drift status: {'stale' if report.is_stale else 'current'}",
        f"Stale: {_yes_no(report.is_stale)}",
        "",
        f"Added files: {len(report.added)}",
        f"Modified files: {len(report.modified)}",
        f"Removed files: {len(report.removed)}",
    ]

    if show_unchanged:
        lines.append(f"Unchanged files: {len(report.unchanged)}")

    _append_drift_group(lines, "Added", report.added)
    _append_drift_group(lines, "Modified", report.modified)
    _append_drift_group(lines, "Removed", report.removed)

    if show_unchanged:
        _append_drift_group(lines, "Unchanged", report.unchanged)

    return "\n".join(lines).rstrip() + "\n"


def _append_drift_group(lines: list[str], title: str, items: list[FileDrift]) -> None:
    if not items:
        return

    lines.append("")
    lines.append(f"{title}:")
    lines.extend(f"- {item.relative_path}" for item in items)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"


if __name__ == "__main__":
    raise SystemExit(main())
