"""Write one pass out as a review bundle: the verdicts *and* the evidence.

`evals/results.json` records what the pipeline decided. That is enough to score
accuracy and not enough to review a proposal, because a reviewer being asked to
approve "write a new page" needs the thing the decision was made against -- which
passages were shortlisted, what they scored, and which of them the model was
actually shown before it said no. Without that, review degrades to trusting the
label, which is the failure mode this whole project is about.

So this exports the evidence alongside the answer, as one JSON file for a UI to
read. Two properties matter and both come from the same choice of implementation:

*Nothing here re-derives a decision.* The pipeline runs unmodified. The client is
wrapped in a recorder that keeps every prompt and reply, and afterwards each
shortlisted passage is matched to its verdict by reconstructing the exact prompt
string and looking it up. Attribution is therefore exact rather than inferred --
which matters because `first_covering` stops at the first affirmative, so a
passage's verdict cannot be worked out from the outcome alone once a page has more
than one section in the shortlist.

*A passage with no recorded verdict says so.* `not_examined` is a distinct value
from `not_answered`. The shortlist is longer than the number of calls whenever the
walk stopped early, and painting those as rejections would show a reviewer five
sections the model rejected when it only ever saw two.

Usage:
    python evals/export_review.py --out evals/review.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from guide_gap import entail
from guide_gap.bedrock import DRAFT_MODEL, EMBED_MODEL, Bedrock
from guide_gap.cache import EmbeddingCache
from guide_gap.corpus import ROOT, load_pages, load_tickets
from guide_gap.coverage import COVERED, FLOOR, Cause
from guide_gap.pipeline import analyse, build_index

CANDIDATES = 5


class Recorder(Bedrock):
    """A Bedrock client that keeps what it was asked and what came back.

    Subclassed rather than wrapped so that `analyse` is handed the real client and
    runs the real code path. A duck-typed stand-in would work until something
    downstream touched an attribute the stand-in forgot, and the point of this
    script is that the exported bundle is a record of a genuine run.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # prompt -> reply. Keyed by the prompt because the prompt contains the
        # ticket and the passage verbatim, which makes the lookup exact.
        self.replies: dict[str, str] = {}
        self.order: list[str] = []

    def draft(self, system: str, prompt: str, max_tokens: int = 2000) -> str:
        reply = super().draft(system, prompt, max_tokens=max_tokens)
        self.replies[prompt] = reply
        self.order.append(prompt)
        return reply


@dataclass
class Bundle:
    """The exported shape. Field names are snake_case to match the producer.

    Deliberately not camelCased on the way out. The consumer is a TypeScript app,
    but renaming here would put a translation layer between two representations of
    the same run, and the first time they disagreed the question "which one is
    wrong" would have no cheap answer.
    """

    generated: dict = field(default_factory=dict)
    run: dict = field(default_factory=dict)
    pages: list = field(default_factory=list)
    tickets: list = field(default_factory=list)
    clusters: list = field(default_factory=list)


def _git_revision() -> str:
    """The commit this bundle was produced from, or "unknown" outside a checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "evals" / "review.json")
    parser.add_argument(
        "--cache",
        type=Path,
        default=ROOT / ".cache" / "embeddings.json",
        help="Embedding cache. Warm, this run only pays for the judgement calls.",
    )
    args = parser.parse_args()

    pages = load_pages()
    tickets = load_tickets()
    by_id = {t.ticket_id: t for t in tickets}

    client = Recorder()
    cache = EmbeddingCache(args.cache, model=EMBED_MODEL)

    print(f"{len(tickets)} tickets, {len(pages)} pages -- this takes a few minutes")
    result = analyse(
        client,
        pages,
        tickets,
        cache=cache,
        candidates=CANDIDATES,
        progress=lambda line: print(f"  {line}"),
    )
    cache.save()

    # Rebuilt rather than returned by `analyse`, because the shortlist is an
    # internal of the coverage loop. Free: every vector needed is now cached.
    index = build_index(client, pages, cache)

    def verdict_for(ticket_text: str, passage: str) -> str:
        prompt = entail.PROMPT.format(ticket=ticket_text.strip(), passage=passage.strip())
        if prompt not in client.replies:
            return "not_examined"
        reply = client.replies[prompt].strip()
        if reply.startswith(entail.NO):
            return "not_answered"
        if reply.startswith(entail.YES):
            return "answered"
        # A reply that is neither word is its own category. Downstream it counts as
        # "not answered", but showing it as a rejection would hide a broken prompt
        # behind a plausible-looking screen.
        return "unparsed"

    exported_tickets = []
    for item in result.classifications:
        ticket = by_id[item.ticket_id]
        vector = cache.get(ticket.text)
        candidates = []
        if vector is not None:
            for match in index.search_passages(vector, k=CANDIDATES):
                page_title, _, heading = match.title.partition(" / ")
                candidates.append(
                    {
                        "page_id": match.page_id,
                        "page_title": page_title,
                        "heading": heading,
                        "score": round(match.score, 4),
                        "text": match.text,
                        "verdict": verdict_for(ticket.text, match.text),
                    }
                )
        exported_tickets.append(
            {
                "ticket_id": ticket.ticket_id,
                "text": ticket.text.strip(),
                "cause": item.cause.value,
                "reason": item.reason,
                "decided_on": item.decided_on,
                "needs_human": ticket.needs_human,
                # The hand-written label, carried so the UI can show where the
                # pipeline and a human already disagree. Written before any model
                # ran; see fixtures/tickets.yaml.
                "expect_cause": ticket.expect_cause,
                "expect_page": ticket.expect_page,
                "label_note": ticket.note,
                "top_score": round(item.best.score, 4) if item.best else None,
                "candidates": candidates,
            }
        )

    drafts_by_key = {tuple(d.ticket_ids): d for d in result.drafts}
    exported_clusters = []
    for cluster in result.clusters:
        draft = drafts_by_key.get(tuple(cluster.ticket_ids))
        exported_clusters.append(
            {
                "ticket_ids": cluster.ticket_ids,
                "nearest_page": cluster.nearest_page,
                "draft": None
                if draft is None
                else {
                    "title": draft.title,
                    "body": draft.body,
                    "source_page": draft.source_page,
                    "open_questions": draft.open_questions,
                    "well_formed": draft.well_formed,
                },
            }
        )

    gaps = sum(1 for c in result.classifications if c.cause is Cause.CONTENT_GAP)
    bundle = Bundle(
        generated={
            "at": datetime.now(UTC).isoformat(timespec="seconds"),
            "revision": _git_revision(),
            "embed_model": EMBED_MODEL,
            "draft_model": DRAFT_MODEL,
            "floor": FLOOR,
            "covered_threshold_baseline_only": COVERED,
            "candidates_per_ticket": CANDIDATES,
        },
        run={
            "tickets": len(tickets),
            "pages": len(pages),
            "sections": len(index),
            "content_gaps": gaps,
            "clusters": len(result.clusters),
            "drafts": len(result.drafts),
            "coverage_checks": result.entail_calls,
            "pair_checks": result.pair_calls,
            "unparsed": result.entail_unparsed,
            "truncated": client.usage.truncated,
            "input_tokens": client.usage.input_tokens,
            "output_tokens": client.usage.output_tokens,
            "seconds": round(client.usage.seconds, 1),
            "errors": client.usage.errors,
        },
        pages=[
            {
                "page_id": page.page_id,
                "title": page.title,
                "keywords": page.keywords,
                "sections": [
                    {"heading": s.heading, "text": s.text} for s in page.sections()
                ],
            }
            for page in pages
        ],
        tickets=exported_tickets,
        clusters=exported_clusters,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(bundle.__dict__, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {args.out} ({args.out.stat().st_size // 1024} KB)")
    print(
        f"{result.entail_calls} coverage checks, {result.pair_calls} pair checks, "
        f"{client.usage.truncated} truncated, {client.usage.input_tokens} in / "
        f"{client.usage.output_tokens} out"
    )


if __name__ == "__main__":
    main()
