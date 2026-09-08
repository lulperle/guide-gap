"""Scoring, and the reason the two error directions are counted separately."""

from __future__ import annotations

import pytest

from guide_gap.corpus import Ticket
from guide_gap.coverage import Cause, Classification
from guide_gap.retrieve import Match
from guide_gap.score import format_score, score_run


def ticket(ticket_id: str, cause: str) -> Ticket:
    return Ticket(ticket_id, "本文", cause, needs_human=cause == "not_deflectable")


def predicted(ticket_id: str, cause: Cause) -> Classification:
    return Classification(ticket_id, cause, Match("p", "P", 0.5, "t"), "")


def test_a_perfect_run_scores_one():
    tickets = [ticket("t1", "content_gap"), ticket("t2", "findability_gap")]
    got = [predicted("t1", Cause.CONTENT_GAP), predicted("t2", Cause.FINDABILITY_GAP)]
    score = score_run(tickets, got)
    assert score.accuracy == pytest.approx(1.0)
    assert score.wrong_ids == []


def test_the_two_error_directions_are_counted_apart():
    """Two runs can score the same and mean opposite things.

    Proposing a page that already exists wastes a reviewer's time. Declaring an
    unwritten topic covered removes it from the report and nobody looks again.
    """
    tickets = [ticket("t1", "findability_gap"), ticket("t2", "content_gap")]
    got = [predicted("t1", Cause.CONTENT_GAP), predicted("t2", Cause.FINDABILITY_GAP)]
    score = score_run(tickets, got)

    assert score.accuracy == pytest.approx(0.0)
    assert score.missed_existing_page == 1
    assert score.hid_a_real_gap == 1
    assert score.wrong_ids == ["t1", "t2"]


def test_an_empty_run_scores_zero_rather_than_dividing_by_zero():
    assert score_run([], []).accuracy == 0.0


def test_the_formatted_score_labels_which_direction_is_costly():
    tickets = [ticket("t1", "content_gap")]
    text = format_score(score_run(tickets, [predicted("t1", Cause.FINDABILITY_GAP)]))
    assert "hid a real gap          1" in text
    assert "costly" in text
