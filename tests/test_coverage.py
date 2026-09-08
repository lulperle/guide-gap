"""The three-way call, and the order the checks happen in."""

from __future__ import annotations

from guide_gap.coverage import Cause, classify
from guide_gap.retrieve import Match

CLOSE = [Match("mfa", "MFA", 0.62, "端末交換の手順")]
FAR = [Match("mfa", "MFA", 0.05, "端末交換の手順")]


def test_needs_human_wins_over_a_very_good_match():
    """A page about permissions cannot grant a permission.

    Ordering this check after the coverage test is the mistake that makes a
    self-service rate look better than it is.
    """
    item = classify("t1", CLOSE, needs_human=True, answered=True)
    assert item.cause is Cause.NOT_DEFLECTABLE
    assert not item.deflectable
    # The nearest page is still reported, so a reviewer can see what it matched.
    assert item.best is not None


def test_nothing_on_the_topic_is_a_content_gap():
    item = classify("t1", FAR, needs_human=False)
    assert item.cause is Cause.CONTENT_GAP
    assert "nothing in the guide" in item.reason


def test_an_empty_corpus_is_a_content_gap_not_a_crash():
    item = classify("t1", [], needs_human=False)
    assert item.cause is Cause.CONTENT_GAP
    assert item.best is None


def test_an_affirmative_verdict_records_the_page_that_answered():
    """Not the top hit. The shortlist walk means the answer can rank second."""
    matches = [Match("incident", "障害", 0.55, "x"), Match("account", "払出", 0.51, "y")]
    item = classify("t1", matches, needs_human=False, answered=True, answered_page="account")
    assert item.cause is Cause.FINDABILITY_GAP
    assert item.decided_on == "account"
    assert item.best.page_id == "incident"


def test_a_negative_verdict_is_a_content_gap_that_names_the_adjacent_page():
    item = classify("t1", CLOSE, needs_human=False, answered=False)
    assert item.cause is Cause.CONTENT_GAP
    # "extend this page" rather than "write a page from nothing" -- and the only
    # grounding the drafting stage gets.
    assert "mfa" in item.reason
    assert item.decided_on is None


def test_no_verdict_falls_back_to_the_threshold():
    """The baseline path, kept so the entailment claim has something to beat."""
    above = classify("t1", CLOSE, needs_human=False, covered=0.45)
    below = classify("t2", [Match("mfa", "MFA", 0.30, "x")], needs_human=False, covered=0.45)
    assert above.cause is Cause.FINDABILITY_GAP
    assert "threshold only" in above.reason
    assert below.cause is Cause.CONTENT_GAP


def test_the_floor_is_checked_before_the_verdict_is_trusted():
    """Below the floor, no check should have run -- and if one did, it is ignored."""
    item = classify("t1", FAR, needs_human=False, answered=True)
    assert item.cause is Cause.CONTENT_GAP
