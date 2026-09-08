"""The numbers this tool exists to move.

Two rules govern everything here, and both are about denominators.

First, the self-service rate is measured against tickets a document *could* have
prevented, not against every ticket. A queue full of access requests will never be
deflected by writing pages, and dividing by it produces a target nobody can reach,
which is how a metric stops being used. Both figures are reported, so the honest
one and the flattering one are visible side by side.

Second, the deflectable tickets are split by cause -- content versus findability --
because that split is the only part of this report that tells anybody what to do
next. A single "70% deflectable" number is a statement with no owner.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .coverage import Cause, Classification


@dataclass
class Report:
    """One batch, summarised. Every field is a count of something observed."""

    total: int = 0
    content_gap: int = 0
    findability_gap: int = 0
    not_deflectable: int = 0
    pages_proposed: int = 0
    tickets_behind_proposals: int = 0
    per_page_findability: dict[str, int] = field(default_factory=dict)

    @property
    def deflectable(self) -> int:
        return self.content_gap + self.findability_gap

    @property
    def deflectable_share(self) -> float:
        """Share of the whole queue a better guide could address."""
        return self.deflectable / self.total if self.total else 0.0

    @property
    def findability_share_of_deflectable(self) -> float:
        """Of the addressable tickets, how many are a search problem.

        The figure that decides where the effort goes. High here means the guide
        already contains the answers and the work is titles, search terms and
        cross-links -- writing more pages would make it worse.
        """
        return self.findability_gap / self.deflectable if self.deflectable else 0.0

    @property
    def coverage_of_proposals(self) -> float:
        """Share of content-gap tickets that the proposed pages would cover.

        Below 1.0 by design: single-ticket topics are excluded from drafting, so
        this shows how much of the gap is long tail rather than pretending the
        drafts closed all of it.
        """
        return (
            self.tickets_behind_proposals / self.content_gap if self.content_gap else 0.0
        )


def summarise(
    classifications: list[Classification],
    *,
    pages_proposed: int = 0,
    tickets_behind_proposals: int = 0,
) -> Report:
    """Aggregate one batch of classifications."""
    report = Report(
        total=len(classifications),
        pages_proposed=pages_proposed,
        tickets_behind_proposals=tickets_behind_proposals,
    )

    for item in classifications:
        if item.cause is Cause.CONTENT_GAP:
            report.content_gap += 1
        elif item.cause is Cause.FINDABILITY_GAP:
            report.findability_gap += 1
            # Which existing page keeps being missed. This is the actionable half
            # of the findability number: "page X answered 6 tickets nobody found"
            # names the page to fix, where a percentage names nothing.
            #
            # `decided_on` rather than the top hit, because the page that answered
            # is not always the page that ranked first, and the whole value of this
            # list is that it sends someone to the right page.
            key = item.decided_on or (item.best.page_id if item.best else None)
            if key:
                report.per_page_findability[key] = (
                    report.per_page_findability.get(key, 0) + 1
                )
        else:
            report.not_deflectable += 1

    return report


def format_report(report: Report) -> str:
    """The report as a human reads it, with both denominators shown."""
    lines = [
        f"tickets analysed          {report.total}",
        f"  content gap             {report.content_gap}  (a page is missing)",
        f"  findability gap         {report.findability_gap}  (the page exists)",
        f"  not deflectable         {report.not_deflectable}  (needs a human)",
        "",
        f"deflectable share         {report.deflectable_share:.0%} of all tickets",
        f"  of which findability    {report.findability_share_of_deflectable:.0%}",
        "",
        f"pages proposed            {report.pages_proposed}",
        f"  tickets they cover      {report.tickets_behind_proposals}"
        f" of {report.content_gap} content-gap tickets"
        f" ({report.coverage_of_proposals:.0%})",
    ]

    if report.per_page_findability:
        lines.append("")
        lines.append("existing pages nobody found:")
        for page, count in sorted(
            report.per_page_findability.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            lines.append(f"  {count:>3}  {page}")

    return "\n".join(lines)
