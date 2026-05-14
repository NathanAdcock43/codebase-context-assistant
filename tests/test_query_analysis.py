from __future__ import annotations

"""
ROLE: Verify deterministic query anchor extraction.
LAYER: tests
FLOW: query_anchor_analysis_validation

INPUTS:
- sample developer questions
- sample pasted ticket descriptions
- path-like source references
- issue keys
- database identifiers
- code identifiers
- quoted UI phrases
- title-style UI phrases

OUTPUTS:
- QueryAnchors behavior assertions
- retrieval query enrichment assertions
- path normalization assertions
- issue key extraction assertions
- database identifier extraction assertions
- code identifier extraction assertions
- UI phrase extraction assertions
- domain-neutral heuristic assertions

UPSTREAM:
- query analysis module
- future ticket-aware retrieval flow
- future query expansion flow

DOWNSTREAM:
- local test runs
- future CI guardrails
- retrieval planning tests
- system map review

OWNS:
- deterministic query anchor tests
- deterministic retrieval query enrichment tests
- fuzzy ticket anchor tests
- source path normalization tests
- strong-anchor detection tests

DOES_NOT_OWN:
- vector store ranking tests
- retrieval sufficiency tests
- ask workflow tests
- API route tests
- LLM query expansion tests
- ticket system integration tests

SIDE_EFFECTS:
- none

STATE:
  reads:
    - in-memory query strings
  writes:
    - none

NOTES:
- These tests protect the deterministic layer before any optional model-based query expansion is added.
- The query analysis module should extract anchors, not infer implementation facts.
- Domain-specific ticket terms may appear in test samples, but not in production heuristic lists.
"""

from pathlib import Path

from code_context.query_analysis import build_retrieval_query, extract_query_anchors, extract_query_paths


def test_extract_query_paths_normalizes_source_paths() -> None:
    paths = extract_query_paths("Explain ./src/code_context/api.py and src\\code_context\\ask.py.")

    assert paths == frozenset(
        {
            "src/code_context/api.py",
            "src/code_context/ask.py",
        }
    )


def test_extract_query_anchors_extracts_database_ticket_identifiers() -> None:
    ticket_text = """
    ASCENDER Maintenance AMP-14751
    Postgres - Liquibase - BFN_FSCL_YR_END_OPT add new column for CREATE_CLASS_1_4

    Alter table BFN_FSCL_YR_END_OPT
    Add new column for CREATE_CLASS_1_4 VARCHAR(1) NOT NULL DEFAULT 'N';
    """

    anchors = extract_query_anchors(ticket_text)

    assert anchors.issue_keys == ("AMP-14751",)
    assert anchors.database_identifiers == (
        "BFN_FSCL_YR_END_OPT",
        "CREATE_CLASS_1_4",
    )
    assert "VARCHAR" in anchors.code_identifiers
    assert anchors.has_strong_anchors is True




def test_query_analysis_production_heuristics_do_not_contain_domain_specific_terms() -> None:
    source_text = Path("src/code_context/query_analysis.py").read_text(encoding="utf-8").lower()

    forbidden_terms = (
        "ascender",
        "mainmenu",
        "password",
        "postgres",
        "liquibase",
        "edfi",
        "regional lea",
        "security",
    )

    assert [term for term in forbidden_terms if term in source_text] == []

def test_extract_query_anchors_keeps_phrase_detection_domain_neutral() -> None:
    anchors = extract_query_anchors(
        'Find likely files for "Update Billing Status" and "Review Export Queue".'
    )

    assert anchors.quoted_phrases == ("Review Export Queue", "Update Billing Status")
    assert anchors.has_strong_anchors is True

def test_extract_query_anchors_extracts_password_ticket_phrases() -> None:
    ticket_text = """
    AMP-14943
    Postgres - ASCENDER - MainMenu Login

    For Change Password for New Password and Confirm Password
    for Create new User - Create Password and Confirm Password
    for Change Password - New Password and Confirm Password
    """

    anchors = extract_query_anchors(ticket_text)

    assert anchors.issue_keys == ("AMP-14943",)
    assert "MainMenu" in anchors.code_identifiers
    assert "Change Password" in anchors.title_phrases
    assert "New Password" in anchors.title_phrases
    assert "Confirm Password" in anchors.title_phrases
    assert anchors.has_strong_anchors is True


def test_extract_query_anchors_extracts_quoted_ui_phrases() -> None:
    anchors = extract_query_anchors(
        'Where is "Create Password" displayed near "Confirm Password"?'
    )

    assert anchors.quoted_phrases == ("Confirm Password", "Create Password")
    assert anchors.has_strong_anchors is True



def test_build_retrieval_query_returns_stripped_query_when_no_anchors_exist() -> None:
    retrieval_query = build_retrieval_query("  Where does this come from?  ")

    assert retrieval_query == "Where does this come from?"


def test_build_retrieval_query_repeats_database_identifiers_for_ticket_search() -> None:
    ticket_text = """
    AMP-14751
    Postgres - Liquibase - BFN_FSCL_YR_END_OPT add new column for CREATE_CLASS_1_4
    Add new column for CREATE_CLASS_1_4 VARCHAR(1) NOT NULL DEFAULT 'N';
    """

    retrieval_query = build_retrieval_query(ticket_text)

    assert retrieval_query.splitlines()[0] == "AMP-14751"
    assert retrieval_query.count("BFN_FSCL_YR_END_OPT") >= 3
    assert retrieval_query.count("CREATE_CLASS_1_4") >= 4
    assert "AMP-14751" in retrieval_query


def test_build_retrieval_query_includes_ui_phrase_anchors() -> None:
    retrieval_query = build_retrieval_query(
        'Where is "Create Password" displayed near "Confirm Password"?'
    )

    assert retrieval_query.count("Create Password") >= 2
    assert retrieval_query.count("Confirm Password") >= 2


def test_extract_query_anchors_handles_single_word_title_phrase_candidate() -> None:
    anchors = extract_query_anchors("Menu")

    assert anchors.title_phrases == ()

def test_extract_query_anchors_marks_plain_fuzzy_question_as_weak() -> None:
    anchors = extract_query_anchors("Where does this come from?")

    assert anchors.paths == ()
    assert anchors.issue_keys == ()
    assert anchors.database_identifiers == ()
    assert anchors.code_identifiers == ()
    assert anchors.quoted_phrases == ()
    assert anchors.has_strong_anchors is False
