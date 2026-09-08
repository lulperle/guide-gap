"""Lambda entry point. Glue only -- everything decidable lives in `guide_gap`.

The pattern this follows: the handler moves bytes and the library makes decisions.
Nothing here does anything that a test would need Lambda to observe, which is why
the test suite covers the pipeline and this file gets a smoke test at most.

Layout in the bucket:

    guide/*.md          the guide, exported from wherever it is authored
    input/tickets.yaml  the batch to analyse
    cache/embeddings.json
    runs/<stamp>/report.txt, drafts/*.md, result.json

Drafts land in S3 and stop there. Nothing in this stack writes to the guide -- a
generated page reaching users without a person reading it first is the outcome the
whole drafting prompt is built to avoid, and an automated commit would make that
one IAM grant away.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

import boto3

from guide_gap.bedrock import Bedrock
from guide_gap.corpus import load_pages, load_tickets
from guide_gap.metrics import format_report
from guide_gap.pipeline import analyse, open_cache
from guide_gap.score import format_score, score_run

BUCKET = os.environ["ARTIFACT_BUCKET"]
MAX_TICKETS = int(os.environ.get("MAX_TICKETS_PER_RUN", "500"))

WORK = Path("/tmp/guide-gap")
s3 = boto3.client("s3")


def run(event: dict, context) -> dict:
    """One scheduled pass. Returns the summary that ends up in the logs."""
    stamp = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    guide_dir = _download_prefix("guide/", WORK / "guide")
    _download("input/tickets.yaml", WORK / "tickets.yaml")
    cache_path = WORK / "embeddings.json"
    _download("cache/embeddings.json", cache_path, required=False)

    pages = load_pages(guide_dir)
    tickets = load_tickets(WORK / "tickets.yaml")

    # Bounded rather than truncated silently. A batch over the cap is a signal that
    # this should not be a Lambda any more, so it is reported in the result instead
    # of showing up as a timeout with no output.
    skipped = max(0, len(tickets) - MAX_TICKETS)
    tickets = tickets[:MAX_TICKETS]

    client = Bedrock()
    cache = open_cache(cache_path)
    result = analyse(client, pages, tickets, cache=cache)
    cache.save()

    prefix = f"runs/{stamp}"
    report = "\n\n".join(
        [
            format_score(score_run(tickets, result.classifications)),
            format_report(result.report),
        ]
    )
    _upload(f"{prefix}/report.txt", report)
    for draft in result.drafts:
        _upload(f"{prefix}/drafts/{'-'.join(draft.ticket_ids)}.md", draft.body)

    summary = {
        "stamp": stamp,
        "tickets": len(tickets),
        "skipped_over_cap": skipped,
        "content_gap": result.report.content_gap,
        "findability_gap": result.report.findability_gap,
        "not_deflectable": result.report.not_deflectable,
        "pages_proposed": result.report.pages_proposed,
        "coverage_checks": result.entail_calls,
        "unparsed_replies": result.entail_unparsed,
        "truncated_replies": client.usage.truncated,
        "errors": client.usage.errors,
    }
    _upload(f"{prefix}/result.json", json.dumps(summary, ensure_ascii=False, indent=2))
    _upload("cache/embeddings.json", cache_path.read_text(encoding="utf-8"))
    return summary


def _download_prefix(prefix: str, into: Path) -> Path:
    into.mkdir(parents=True, exist_ok=True)
    pages = s3.get_paginator("list_objects_v2")
    for page in pages.paginate(Bucket=BUCKET, Prefix=prefix):
        for item in page.get("Contents", []):
            name = item["Key"].removeprefix(prefix)
            if name:
                _download(item["Key"], into / name)
    return into


def _download(key: str, to: Path, *, required: bool = True) -> None:
    to.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3.download_file(BUCKET, key, str(to))
    except s3.exceptions.ClientError:
        # A missing cache is normal on the first run and costs money, not
        # correctness. A missing ticket export is not, and should stop the run.
        if required:
            raise


def _upload(key: str, body: str) -> None:
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=body.encode("utf-8"),
        ContentType="text/plain; charset=utf-8",
    )
