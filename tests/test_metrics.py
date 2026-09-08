"""Denominators, mostly. That is where a support metric goes wrong."""

from __future__ import annotations

import pytest

from guide_gap.coverage import Cause, Classification
from guide_gap.metrics import format_report, summarise
from guide_gap.retrieve import Match


def item(ticket_id: str, cause: Cause, page: str = "mfa", decided: str | None = None):
    return Classification(ticket_id, cause, Match(page, page, 0.5, "t"), "", decided)


BATCH = [
    item("t1", Cause.FINDABILITY_GAP, "mfa", "mfa"),
    item("t2", Cause.FINDABILITY_GAP, "mfa", "mfa"),
    item("t3", Cause.CONTENT_GAP, "cost"),
    item("t4", Cause.NOT_DEFLECTABLE, "account"),
]


def test_the_self_service_rate_is_measured_against_deflectable_tickets():
    report = summarise(BATCH)
    assert report.total == 4
    assert report.deflectable == 3
    # Both denominators are reported, so the honest one and the flattering one are
    # visible side by side: 3 of 4 tickets are addressable, and of those, 2 are a
    # search problem rather than a writing problem.
    assert report.deflectable_share == pytest.approx(0.75)
    assert report.findability_share_of_deflectable == pytest.approx(2 / 3)


def test_an_empty_batch_reports_zero_rather_than_dividing_by_zero():
    report = summarise([])
    assert report.deflectable_share == 0.0
    assert report.findability_share_of_deflectable == 0.0
    assert report.coverage_of_proposals == 0.0


def test_findability_is_attributed_to_the_page_that_answered():
    """`decided_on`, not the top hit -- the whole value of the list is sending
    somebody to the right page."""
    batch = [item("t1", Cause.FINDABILITY_GAP, page="incident", decided="account")]
    report = summarise(batch)
    assert report.per_page_findability == {"account": 1}


def test_coverage_of_proposals_shows_the_long_tail_it_left_behind():
    report = summarise(
        BATCH + [item("t5", Cause.CONTENT_GAP, "cost")],
        pages_proposed=1,
        tickets_behind_proposals=2,
    )
    assert report.content_gap == 2
    assert report.coverage_of_proposals == pytest.approx(1.0)

    partial = summarise(BATCH, pages_proposed=0, tickets_behind_proposals=0)
    # One content gap, no page proposed: the report says so rather than claiming
    # the drafts closed everything.
    assert partial.coverage_of_proposals == 0.0


def test_the_formatted_report_names_the_pages_nobody_found():
    text = format_report(summarise(BATCH, pages_proposed=1, tickets_behind_proposals=1))
    assert "existing pages nobody found" in text
    assert "mfa" in text
    assert "of all tickets" in text
