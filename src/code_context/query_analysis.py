from __future__ import annotations

"""
ROLE: Extract deterministic retrieval anchors from developer questions and pasted tickets.
LAYER: retrieval
FLOW: query_anchor_analysis

INPUTS:
- developer question text
- pasted ticket descriptions
- source path references
- code/database identifier text
- quoted UI or workflow phrases

OUTPUTS:
- QueryAnchors records
- enriched deterministic retrieval query text
- normalized source path anchors
- issue key anchors
- database identifier anchors
- code identifier anchors
- quoted phrase anchors
- title phrase anchors

UPSTREAM:
- future retrieval planning service
- future ticket-aware retrieval flow
- local tests
- future CLI and API ask workflows

DOWNSTREAM:
- vector retrieval query construction
- retrieval query enrichment
- retrieval sufficiency checks
- future query expansion service
- future lightweight LLM query rewrite service

OWNS:
- deterministic query anchor extraction
- deterministic retrieval query enrichment
- path-like text normalization
- issue key detection
- database identifier detection
- code-like identifier detection
- quoted phrase extraction
- conservative title phrase extraction

DOES_NOT_OWN:
- vector scoring
- source chunk search
- retrieval sufficiency decisions
- LLM-based query expansion
- ticket system integration
- answer generation
- API routing
- CLI argument parsing

SIDE_EFFECTS:
- none

STATE:
  reads:
    - none
  writes:
    - none

NOTES:
- This module is intentionally deterministic.
- It gives fuzzy ticket retrieval a safer first pass before any model-based query expansion.
- It should extract anchors, not infer facts about the codebase.
- Query enrichment repeats extracted anchors to improve recall without adding guessed terms.
"""

import re
from dataclasses import dataclass


PATH_LIKE_PATTERN = re.compile(
    r"(?:[A-Za-z0-9_.-]+[\\/])+[A-Za-z0-9_.-]+\.[A-Za-z0-9_]+"
)
ISSUE_KEY_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]+-\d+\b")
DATABASE_IDENTIFIER_PATTERN = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")
IDENTIFIER_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
QUOTED_PHRASE_PATTERN = re.compile(r"['\"]([^'\"]{3,80})['\"]")
TITLE_PHRASE_PATTERN = re.compile(
    r"\b[A-Z][A-Za-z0-9]+(?:\s+[A-Z][A-Za-z0-9]+){1,5}\b"
)

COMMON_TITLE_PHRASES = frozenset(
    {
        "Active Development",
        "Add Comment",
        "Affects Version",
        "Create Mode",
        "Details Type",
        "Ed Fi Knowledge Hub",
        "Field Tab",
        "Fix Version",
        "Issue Attributes",
        "No Release Notes",
        "Regional Lea Impact",
        "Regional Lea Reach",
        "Share This Issue",
        "Source Projects",
        "System Category",
    }
)

COMMON_IDENTIFIER_WORDS = frozenset(
    {
        "add",
        "and",
        "comment",
        "details",
        "edit",
        "for",
        "from",
        "high",
        "low",
        "medium",
        "more",
        "none",
        "not",
        "option",
        "postgres",
        "priority",
        "share",
        "sub",
        "task",
        "the",
        "this",
        "to",
        "type",
    }
)


@dataclass(frozen=True)
class QueryAnchors:
    """Deterministic anchors extracted from a developer query or pasted ticket."""

    paths: tuple[str, ...]
    issue_keys: tuple[str, ...]
    database_identifiers: tuple[str, ...]
    code_identifiers: tuple[str, ...]
    quoted_phrases: tuple[str, ...]
    title_phrases: tuple[str, ...]

    @property
    def has_strong_anchors(self) -> bool:
        return bool(
            self.paths
            or self.issue_keys
            or self.database_identifiers
            or self.code_identifiers
            or self.quoted_phrases
        )


def extract_query_anchors(query: str) -> QueryAnchors:
    """Extract conservative retrieval anchors from free-form query text."""

    return QueryAnchors(
        paths=_sorted_unique(_normalize_path(match) for match in PATH_LIKE_PATTERN.findall(query)),
        issue_keys=_sorted_unique(ISSUE_KEY_PATTERN.findall(query)),
        database_identifiers=_sorted_unique(DATABASE_IDENTIFIER_PATTERN.findall(query)),
        code_identifiers=_sorted_unique(_extract_code_identifiers(query)),
        quoted_phrases=_sorted_unique(_normalize_phrase(match) for match in QUOTED_PHRASE_PATTERN.findall(query)),
        title_phrases=_sorted_unique(_extract_title_phrases(query)),
    )


def extract_query_paths(query: str) -> frozenset[str]:
    """Extract normalized source paths from query text."""

    return frozenset(_normalize_path(match) for match in PATH_LIKE_PATTERN.findall(query))


def build_retrieval_query(query: str) -> str:
    """Build deterministic search text from a query and its extracted anchors."""

    normalized_query = query.strip()
    if not normalized_query:
        return ""

    anchors = extract_query_anchors(normalized_query)
    anchor_lines = _anchor_lines(anchors)

    if not anchor_lines:
        return normalized_query

    return "\n".join([normalized_query, *anchor_lines])


def _anchor_lines(anchors: QueryAnchors) -> tuple[str, ...]:
    lines: list[str] = []

    lines.extend(anchors.paths)
    lines.extend(anchors.database_identifiers)
    lines.extend(anchors.database_identifiers)
    lines.extend(anchors.code_identifiers)
    lines.extend(anchors.quoted_phrases)
    lines.extend(anchors.title_phrases)
    lines.extend(anchors.issue_keys)

    return tuple(line for line in lines if line)


def _extract_code_identifiers(query: str) -> tuple[str, ...]:
    identifiers: list[str] = []

    for token in IDENTIFIER_PATTERN.findall(query):
        normalized = token.lower()
        if normalized in COMMON_IDENTIFIER_WORDS:
            continue

        if _looks_like_code_identifier(token):
            identifiers.append(token)

    return tuple(identifiers)


def _looks_like_code_identifier(token: str) -> bool:
    if "_" in token:
        return True

    if any(character.isdigit() for character in token):
        return True

    if token.isupper() and len(token) > 1:
        return True

    return any(character.isupper() for character in token[1:])


def _extract_title_phrases(query: str) -> tuple[str, ...]:
    phrases: list[str] = []

    for match in TITLE_PHRASE_PATTERN.findall(query):
        normalized = _normalize_phrase(match)

        if normalized.title() in COMMON_TITLE_PHRASES:
            continue

        if _phrase_has_retrieval_value(normalized):
            phrases.append(normalized)

    return tuple(phrases)


def _phrase_has_retrieval_value(phrase: str) -> bool:
    words = phrase.split()

    if len(words) < 2:
        return False

    return any(
        word.lower() in {"change", "confirm", "create", "login", "new", "password", "user"}
        for word in words
    )


def _normalize_path(path: str) -> str:
    return path.strip().strip("'\"`.,:;()[]{}").replace("\\", "/").lstrip("./").lower()


def _normalize_phrase(phrase: str) -> str:
    return " ".join(phrase.strip().split())


def _sorted_unique(values) -> tuple[str, ...]:
    return tuple(sorted({value for value in values if value}))
