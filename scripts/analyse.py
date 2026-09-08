"""Run one pass over the ticket batch and print the report.

    python scripts/analyse.py                # classify only, no generation
    python scripts/analyse.py --drafts       # also draft the pages
    python scripts/analyse.py --scores       # per-ticket similarity, for thresholds
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guide_gap.bedrock import Bedrock  # noqa: E402
from guide_gap.metrics import format_report  # noqa: E402
from guide_gap.pipeline import analyse, load_all, open_cache  # noqa: E402
from guide_gap.score import format_score, score_run  # noqa: E402

CACHE = Path(__file__).resolve().parent.parent / ".cache" / "embeddings.json"
DRAFT_DIR = Path(__file__).resolve().parent.parent / "out" / "drafts"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drafts", action="store_true", help="generate draft pages")
    parser.add_argument("--scores", action="store_true", help="print per-ticket scores")
    parser.add_argument("--no-cache", action="store_true", help="ignore the embedding cache")
    parser.add_argument("--write", action="store_true", help="write drafts to out/drafts")
    parser.add_argument("--quiet", action="store_true", help="no progress lines")
    parser.add_argument(
        "--no-entail",
        action="store_true",
        help="decide coverage by similarity threshold only (the baseline)",
    )
    args = parser.parse_args()

    pages, tickets = load_all()
    client = Bedrock()
    cache = None if args.no_cache else open_cache(CACHE)

    result = analyse(
        client,
        pages,
        tickets,
        cache=cache,
        write_drafts=args.drafts,
        entail=not args.no_entail,
        # To stderr, so `... > report.txt` keeps the report clean and the progress
        # still shows on the terminal.
        progress=None if args.quiet else lambda line: print(line, file=sys.stderr),
    )
    if cache:
        cache.save()

    mode = "threshold only" if args.no_entail else "entailment on passages"
    print(f"guide pages {len(pages)}   tickets {len(tickets)}   mode: {mode}")
    print()
    print(format_score(score_run(tickets, result.classifications)))
    print()
    print(format_report(result.report))

    if args.scores:
        print()
        print("per-ticket  (label / predicted / nearest page)")
        want = {t.ticket_id: t.expect_cause for t in tickets}
        for item in result.classifications:
            best = f"{item.best.score:.3f} {item.best.page_id}" if item.best else "-"
            decided = f" -> {item.decided_on}" if item.decided_on else ""
            mark = " " if want[item.ticket_id] == item.cause.value else "x"
            print(
                f"  {mark} {item.ticket_id}  {want[item.ticket_id]:<16}"
                f" {item.cause.value:<16} {best}{decided}"
            )

    print()
    print(f"clusters of content-gap tickets ({len(result.clusters)}):")
    for cluster in result.clusters:
        near = cluster.nearest_page or "-"
        print(f"  {cluster.size}  {','.join(cluster.ticket_ids):<16} near={near}")

    if result.drafts:
        print()
        for draft in result.drafts:
            flag = "" if draft.well_formed else "  [MALFORMED]"
            print(
                f"draft: {draft.title}{flag}\n"
                f"  tickets {','.join(draft.ticket_ids)}"
                f"  open questions {len(draft.open_questions)}"
            )
        if args.write:
            DRAFT_DIR.mkdir(parents=True, exist_ok=True)
            for draft in result.drafts:
                name = "-".join(draft.ticket_ids) + ".md"
                (DRAFT_DIR / name).write_text(draft.body + "\n", encoding="utf-8")
            print(f"\nwritten to {DRAFT_DIR}")

    usage = client.usage
    print()
    print(
        f"usage: embed {usage.embed_calls} calls / {usage.embed_tokens} tokens, "
        f"draft {usage.draft_calls} calls / "
        f"{usage.input_tokens} in / {usage.output_tokens} out, "
        f"{usage.seconds:.1f}s"
    )
    print(
        f"  coverage checks {result.entail_calls}"
        f"  pair checks {result.pair_calls}"
        f"  unparsed replies {result.entail_unparsed}"
        f"  truncated {usage.truncated}"
    )
    for error in usage.errors:
        print(f"  error: {error}")

    return 1 if usage.errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
