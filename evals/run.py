"""Measure the pipeline, repeatedly, and write the raw artifact.

Two rules, both learned the hard way on this project.

**Repeat, because one run is not a measurement.** Claude Sonnet 5 rejects
`temperature`, so there is no way to pin the output; the only honest response is to
run the batch several times and report the spread rather than the best number. A
single 95% and a 95%-on-average-with-one-run-at-77% justify completely different
decisions, and only one of them is worth shipping.

**Score against labels written first.** `fixtures/tickets.yaml` carries the expected
cause for every ticket, written before any of this ran. Comparison is string
equality. No model grades another model's output here -- a judge model would have
happily rated the truncated-to-empty replies as reasonable, and the whole
truncation bug would still be in the repository.

The JSON artifact holds every per-ticket prediction from every run, so any number
in the README can be recomputed from it without a single API call.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from guide_gap.bedrock import DRAFT_MODEL, EMBED_MODEL, Bedrock  # noqa: E402
from guide_gap.pipeline import analyse, load_all, open_cache  # noqa: E402
from guide_gap.score import score_run  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache" / "embeddings.json"
ARTIFACT = ROOT / "evals" / "results.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=3, help="how many passes")
    parser.add_argument(
        "--no-entail",
        action="store_true",
        help="measure the similarity-threshold baseline instead",
    )
    parser.add_argument(
        "--drafts",
        action="store_true",
        help="also generate drafts and check they are well formed",
    )
    parser.add_argument("--out", type=Path, default=ARTIFACT)
    args = parser.parse_args()

    pages, tickets = load_all()
    cache = open_cache(CACHE)
    runs = []

    for number in range(1, args.repeat + 1):
        client = Bedrock()
        started = time.monotonic()
        result = analyse(
            client,
            pages,
            tickets,
            cache=cache,
            entail=not args.no_entail,
            write_drafts=args.drafts,
            # `number` bound as a default: a closure over the loop variable would
            # label every line with the last run once this loop is ever parallelised.
            progress=lambda line, n=number: print(f"  run {n}: {line}", file=sys.stderr),
        )
        cache.save()
        score = score_run(tickets, result.classifications)

        runs.append(
            {
                "run": number,
                "seconds": round(time.monotonic() - started, 1),
                "accuracy": score.accuracy,
                "correct": score.correct,
                "total": score.total,
                "missed_existing_page": score.missed_existing_page,
                "hid_a_real_gap": score.hid_a_real_gap,
                "wrong": score.wrong_ids,
                "predictions": {
                    c.ticket_id: c.cause.value for c in result.classifications
                },
                "decided_on": {
                    c.ticket_id: c.decided_on for c in result.classifications
                },
                "clusters": [c.ticket_ids for c in result.clusters],
                "pages_proposed": result.report.pages_proposed,
                "tickets_behind_proposals": result.report.tickets_behind_proposals,
                "drafts": [
                    {
                        "tickets": d.ticket_ids,
                        "title": d.title,
                        "well_formed": d.well_formed,
                        "open_questions": len(d.open_questions),
                    }
                    for d in result.drafts
                ],
                "coverage_checks": result.entail_calls,
                "pair_checks": result.pair_calls,
                "unparsed": result.entail_unparsed,
                "truncated": client.usage.truncated,
                "input_tokens": client.usage.input_tokens,
                "output_tokens": client.usage.output_tokens,
                "errors": client.usage.errors,
            }
        )
        print(
            f"run {number}: {score.correct}/{score.total} "
            f"({score.accuracy:.0%})  wrong: {', '.join(score.wrong_ids) or '-'}"
        )

    report = summarise_runs(runs, entail=not args.no_entail)
    args.out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print()
    print(format_runs(report))
    print(f"\nartifact: {args.out}")
    return 0


def summarise_runs(runs: list[dict], *, entail: bool) -> dict:
    """Aggregate the passes, keeping every raw run."""
    accuracies = [r["accuracy"] for r in runs]
    # Which tickets were wrong in *some* runs but not all. These are the flaky
    # ones, and they are worth naming separately from the consistently-wrong ones:
    # a ticket that fails half the time is a stability problem, and a ticket that
    # fails every time is a defect. The same accuracy number covers both.
    wrong_counts: dict[str, int] = {}
    for run in runs:
        for ticket in run["wrong"]:
            wrong_counts[ticket] = wrong_counts.get(ticket, 0) + 1

    return {
        "mode": "entailment" if entail else "threshold",
        "embed_model": EMBED_MODEL,
        "draft_model": DRAFT_MODEL,
        "repeats": len(runs),
        "accuracy_min": min(accuracies),
        "accuracy_max": max(accuracies),
        "accuracy_mean": statistics.fmean(accuracies),
        "always_wrong": sorted(t for t, n in wrong_counts.items() if n == len(runs)),
        "sometimes_wrong": sorted(t for t, n in wrong_counts.items() if n < len(runs)),
        "wrong_counts": dict(sorted(wrong_counts.items())),
        "hid_a_real_gap_total": sum(r["hid_a_real_gap"] for r in runs),
        "truncated_total": sum(r["truncated"] for r in runs),
        "unparsed_total": sum(r["unparsed"] for r in runs),
        "runs": runs,
    }


def format_runs(report: dict) -> str:
    total = report["runs"][0]["total"]
    lines = [
        f"mode                {report['mode']}   repeats {report['repeats']}",
        f"accuracy            {report['accuracy_mean'] * total:.1f}/{total} mean"
        f"  (min {report['accuracy_min']:.0%}, max {report['accuracy_max']:.0%})",
        f"always wrong        {', '.join(report['always_wrong']) or '-'}",
        f"flaky               {', '.join(report['sometimes_wrong']) or '-'}",
        f"hid a real gap      {report['hid_a_real_gap_total']} across all runs",
        f"truncated replies   {report['truncated_total']}",
        f"unparsed replies    {report['unparsed_total']}",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
