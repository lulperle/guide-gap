"""Why did this ticket get raised when a guide exists?

The distinction this module exists for, and the reason it is not just a similarity
search: a ticket the guide already answers is a **findability** failure, and a
ticket the guide does not answer is a **content** failure. They look identical in
a ticket queue and they have different fixes and different owners.

    content gap      -> somebody has to write a page
    findability gap  -> the page exists; the title, the search terms, or the
                        cross-links from where the user actually was are wrong

Conflating them produces the worst outcome available: new pages covering ground
the guide already covers, which makes the guide bigger, harder to search, and
therefore worse at the thing that caused the ticket. Volume of new pages is not
progress.

The third bucket matters as much. Some tickets are not deflectable by any
document -- an account-specific permission that only an administrator can grant,
a defect, a request for a decision. Counting those as guide failures makes the
self-service target unreachable and the metric dishonest, and a target nobody can
hit stops being used.

How the content/findability call is made is a measured decision, not a taste one.
Two ways are implemented: a similarity threshold, and an entailment check on the
retrieved passage. The threshold is kept because it is the baseline the second one
has to beat, and because a claim of improvement with nothing to compare against is
just an assertion. `evals/run.py` runs both.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .retrieve import Match


class Cause(StrEnum):
    """Why the ticket happened, in terms of what would stop the next one.

    A string enum so the value is what lands in the JSON artifact and in the
    hand-written labels, and the comparison in `score.py` is the same comparison a
    person makes reading the fixture file.
    """

    CONTENT_GAP = "content_gap"
    FINDABILITY_GAP = "findability_gap"
    NOT_DEFLECTABLE = "not_deflectable"


# Similarity below which nothing is worth checking further.
#
# Not a coverage threshold -- a shortlist floor. Its only job is to skip the
# entailment call when the corpus contains nothing on the topic at all, and it is
# set below the lowest-scoring ticket in the labelled batch that the guide does in
# fact answer (0.24). Set it near where the real decision boundary feels like it
# should be and it starts silently deciding cases instead of filtering them, which
# reintroduces the exact failure the entailment check was added to remove.
FLOOR = 0.20

# Used only by the threshold baseline in the comparison. Placed at the point that
# maximises accuracy on the labelled batch -- which is to say it is fitted to this
# batch and would not survive a different one. That is the finding, not a caveat:
# see the README.
COVERED = 0.45


@dataclass
class Classification:
    """One ticket's diagnosis, with the evidence that produced it."""

    ticket_id: str
    cause: Cause
    best: Match | None
    reason: str
    # Which page the decision was actually made against. Not always `best`: the
    # entailment check walks the shortlist, so the page that answered the ticket
    # can be the second or third result, and reporting the top hit instead would
    # send the reviewer to the wrong page.
    decided_on: str | None = None

    @property
    def deflectable(self) -> bool:
        """Whether a better guide could have prevented this ticket."""
        return self.cause is not Cause.NOT_DEFLECTABLE


def classify(
    ticket_id: str,
    matches: list[Match],
    *,
    needs_human: bool,
    answered: bool | None = None,
    answered_page: str | None = None,
    floor: float = FLOOR,
    covered: float = COVERED,
) -> Classification:
    """Diagnose one ticket.

    Args:
        ticket_id: The ticket being classified.
        matches: Guide passages ranked by similarity, best first, one per page.
        needs_human: Whether the request requires an action no document can
            perform -- granting access, making a decision, fixing a defect. An
            input rather than something inferred here, because it is a property of
            the request, not of the corpus.
        answered: The entailment verdict, when one was obtained. `None` means no
            check was run, and the function falls back to the similarity threshold
            -- the documented degraded mode used by the baseline, not a default to
            rely on.
        answered_page: Which page produced an affirmative verdict.
        floor: Similarity below which the corpus has nothing on the topic.
        covered: Threshold used only in the fallback path.

    Returns:
        A Classification carrying the evidence that justified it.
    """
    # Checked first, and deliberately: a permission request does not become
    # deflectable because a page about permissions happens to score well. The page
    # cannot grant the permission. Ordering this after the coverage test is the
    # mistake that makes a self-service rate look better than it is.
    if needs_human:
        return Classification(
            ticket_id,
            Cause.NOT_DEFLECTABLE,
            matches[0] if matches else None,
            "requires an action no document can perform",
        )

    if not matches:
        return Classification(ticket_id, Cause.CONTENT_GAP, None, "empty guide corpus")

    best = matches[0]

    if best.score < floor:
        return Classification(
            ticket_id,
            Cause.CONTENT_GAP,
            best,
            f"nothing in the guide is on this topic (best {best.score:.2f})",
        )

    if answered is None:
        if best.score >= covered:
            return Classification(
                ticket_id,
                Cause.FINDABILITY_GAP,
                best,
                f"similarity {best.score:.2f} at or above {covered:.2f} (threshold only)",
                best.page_id,
            )
        return Classification(
            ticket_id,
            Cause.CONTENT_GAP,
            best,
            f"similarity {best.score:.2f} below {covered:.2f} (threshold only)",
        )

    if answered:
        page = answered_page or best.page_id
        return Classification(
            ticket_id,
            Cause.FINDABILITY_GAP,
            best,
            f"{page} answers this; the user did not find it",
            page,
        )

    # Retrieved something adjacent but it does not answer. Still a content gap --
    # the answer is not there -- but the nearest page says where the answer belongs,
    # which is the difference between "write a page" and "extend this page", and it
    # is the only grounding the drafting stage gets.
    return Classification(
        ticket_id,
        Cause.CONTENT_GAP,
        best,
        f"nearest page {best.page_id} ({best.score:.2f}) is adjacent, not an answer",
    )
