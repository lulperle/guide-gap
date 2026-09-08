"""Score a run against the hand-written labels.

Deterministic: string equality against `expect_cause` in the fixtures, which was
written before any of this ran. No model grades anything here.

The per-direction breakdown is the point, not the headline percentage. Two runs
can both score 77% and mean opposite things:

    findability -> content   proposes rewriting a page that already exists.
                             Wasteful, but a reviewer sees the duplicate and
                             rejects the draft.
    content -> findability   declares an unwritten topic covered. The gap
                             disappears from the report and nobody ever looks
                             again.

The second is strictly worse and a single accuracy number cannot tell them apart,
so it is reported separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .corpus import Ticket
from .coverage import Classification


@dataclass
class Score:
    """Accuracy of one run, split by how it was wrong."""

    total: int = 0
    correct: int = 0
    # (expected, predicted) -> count
    confusion: dict[tuple[str, str], int] = field(default_factory=dict)
    wrong_ids: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def missed_existing_page(self) -> int:
        """Said "write a page" when the guide already answered."""
        return self.confusion.get(("findability_gap", "content_gap"), 0)

    @property
    def hid_a_real_gap(self) -> int:
        """Said "the guide covers it" when nothing was written. The costly one."""
        return self.confusion.get(("content_gap", "findability_gap"), 0)


def score_run(tickets: list[Ticket], classifications: list[Classification]) -> Score:
    """Compare predictions against labels, ticket by ticket."""
    expected = {t.ticket_id: t.expect_cause for t in tickets}
    result = Score(total=len(classifications))

    for item in classifications:
        want = expected[item.ticket_id]
        got = item.cause.value
        if want == got:
            result.correct += 1
            continue
        result.wrong_ids.append(item.ticket_id)
        key = (want, got)
        result.confusion[key] = result.confusion.get(key, 0) + 1

    return result


def format_score(score: Score) -> str:
    lines = [
        f"accuracy                  {score.correct}/{score.total}"
        f"  ({score.accuracy:.0%})",
        f"  missed an existing page {score.missed_existing_page}"
        "  (wasteful: proposes a duplicate)",
        f"  hid a real gap          {score.hid_a_real_gap}"
        "  (costly: the gap stops being reported)",
    ]
    if score.wrong_ids:
        lines.append(f"  wrong                   {', '.join(score.wrong_ids)}")
    for (want, got), count in sorted(score.confusion.items()):
        lines.append(f"    {want} -> {got}: {count}")
    return "\n".join(lines)
