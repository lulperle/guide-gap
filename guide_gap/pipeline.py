"""One pass: tickets in, a report and some draft pages out.

The order of the stages is the argument of this project.

    embed -> classify (content vs findability vs not deflectable)
          -> cluster ONLY the content gaps
          -> draft ONLY the clusters big enough to justify a page

Clustering everything and drafting the biggest groups is the obvious shape and it
is wrong. The biggest group in a support queue is usually a question the guide
already answers -- that is why it is the biggest group -- so drafting from raw
volume writes a second page about MFA and leaves the real hole untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .bedrock import EMBED_MODEL, Bedrock
from .cache import EmbeddingCache
from .cluster import (
    MIN_CLUSTER_SIZE,
    PAIR_FLOOR,
    Cluster,
    cluster_tickets,
    worth_writing,
)
from .corpus import Page, Ticket, load_pages, load_tickets
from .coverage import COVERED, FLOOR, Cause, Classification, classify
from .draft import Draft, draft_page
from .entail import first_covering, same_question
from .metrics import Report, summarise
from .retrieve import Index


@dataclass
class Result:
    """Everything one pass produced, kept so the eval can score it."""

    classifications: list[Classification] = field(default_factory=list)
    clusters: list[Cluster] = field(default_factory=list)
    drafts: list[Draft] = field(default_factory=list)
    report: Report = field(default_factory=Report)
    entail_calls: int = 0
    entail_unparsed: int = 0
    pair_calls: int = 0
    scores: dict[str, float] = field(default_factory=dict)


def build_index(
    client: Bedrock,
    pages: list[Page],
    cache: EmbeddingCache | None = None,
    *,
    by_section: bool = True,
) -> Index:
    """Embed the guide and make it searchable.

    Args:
        by_section: Index each `##` section separately, a page scoring as its best
            section. On by default because whole-page indexing measurably does not
            work here. The flag stays so the comparison can be re-run rather than
            taken on trust -- `evals/run.py --whole-page` does exactly that.
    """
    if by_section:
        sections = [s for page in pages for s in page.sections()]
        return Index(
            page_ids=[s.page_id for s in sections],
            titles=[f"{s.page_title.splitlines()[0]} / {s.heading}" for s in sections],
            vectors=[_embed(client, s.embedding_text, cache) for s in sections],
            # The passage the coverage decision is made against. Forgetting this
            # argument is not a subtle bug in effect but it is a silent one in
            # appearance: every passage arrives empty, the model correctly answers
            # "no, this does not answer the question", and the whole batch reports
            # as a content gap with no error anywhere. Hence the guard in
            # entail.answers().
            texts=[s.embedding_text for s in sections],
        )

    bodies = [f"{p.title}\n{'、'.join(p.keywords)}\n\n{p.body}" for p in pages]
    return Index(
        page_ids=[p.page_id for p in pages],
        titles=[p.title for p in pages],
        vectors=[_embed(client, body, cache) for body in bodies],
        texts=bodies,
    )


def analyse(
    client: Bedrock,
    pages: list[Page],
    tickets: list[Ticket],
    *,
    cache: EmbeddingCache | None = None,
    covered: float = COVERED,
    floor: float = FLOOR,
    minimum: int = MIN_CLUSTER_SIZE,
    pair_floor: float = PAIR_FLOOR,
    write_drafts: bool = True,
    by_section: bool = True,
    entail: bool = True,
    candidates: int = 5,
    progress: Callable[[str], None] | None = None,
) -> Result:
    """Run the full pass.

    Args:
        write_drafts: Off for threshold work. Classification and clustering are
            deterministic given the embeddings, so with a warm cache the whole
            analysis re-runs for free -- but only if it is not also making
            generation calls. Separating them is what makes the thresholds
            cheap to argue about.
        entail: Decide coverage by asking whether the retrieved passage answers the
            ticket. Off falls back to the similarity threshold, which is the
            baseline the eval compares against.
        candidates: How many shortlisted *passages* the entailment check may look
            at before giving up. Each one is a call, so this is the cost dial.
        progress: Called with a one-line status per ticket. Not decoration: the
            entailment pass is dozens of sequential calls and takes minutes, and a
            batch job that prints nothing for two minutes gets interrupted by
            whoever is watching it, which is exactly what happened the first time
            this ran.
    """
    def note(message: str) -> None:
        if progress:
            progress(message)

    note(f"indexing {len(pages)} guide pages")
    index = build_index(client, pages, cache, by_section=by_section)
    note(f"index ready: {len(index)} sections across {index.pages} pages")
    bodies = {p.page_id: p.body for p in pages}

    classifications = []
    vectors: dict[str, list[float]] = {}
    scores: dict[str, float] = {}
    entail_calls = 0
    unparsed = 0

    for position, ticket in enumerate(tickets, start=1):
        vector = _embed(client, ticket.text, cache)
        vectors[ticket.ticket_id] = vector
        matches = index.search(vector, k=3)
        scores[ticket.ticket_id] = matches[0].score if matches else 0.0

        answered: bool | None = None
        answered_page: str | None = None
        # The check is skipped when it cannot change the answer: a ticket needing a
        # human is not deflectable whatever the guide says, and a corpus with
        # nothing above the floor has nothing to entail from. Skipping those is
        # most of the cost saving, and it is free -- the classification is
        # identical either way.
        if entail and not ticket.needs_human and matches and matches[0].score >= floor:
            # Passages, not pages. The page-level list is for the report; the
            # coverage decision needs the actual sections, including several from
            # one page.
            verdict, calls = first_covering(
                client,
                ticket.text,
                index.search_passages(vector, k=candidates),
                limit=candidates,
            )
            entail_calls += calls
            if verdict is not None:
                answered = verdict.answered
                answered_page = verdict.page_id
                if not verdict.parsed:
                    unparsed += 1

        item = classify(
            ticket.ticket_id,
            matches,
            needs_human=ticket.needs_human,
            answered=answered,
            answered_page=answered_page,
            floor=floor,
            covered=covered,
        )
        classifications.append(item)
        note(
            f"[{position}/{len(tickets)}] {ticket.ticket_id} "
            f"{item.cause.value} ({entail_calls} checks so far)"
        )

    gaps = [c for c in classifications if c.cause is Cause.CONTENT_GAP]
    by_id = {t.ticket_id: t for t in tickets}

    pair_calls = 0

    def confirm(first: str, second: str) -> bool:
        nonlocal pair_calls, unparsed
        pair_calls += 1
        verdict = same_question(client, first, second)
        if not verdict.parsed:
            unparsed += 1
        return verdict.answered

    note(f"clustering {len(gaps)} content-gap tickets")
    clusters = cluster_tickets(
        ticket_ids=[c.ticket_id for c in gaps],
        texts=[by_id[c.ticket_id].text for c in gaps],
        vectors=[vectors[c.ticket_id] for c in gaps],
        # The adjacent page, when there was one. This is where the new section
        # should probably live, and it is also the only grounding the drafting
        # stage gets -- so a cluster with no adjacent page yields a draft that is
        # mostly open questions, which is the correct outcome rather than a
        # degraded one.
        nearest_pages=[c.best.page_id if c.best else None for c in gaps],
        confirm=confirm if entail else None,
        floor=pair_floor,
    )
    note(f"clusters: {len(clusters)} from {pair_calls} pair checks")

    proposals = worth_writing(clusters, minimum=minimum)
    drafts = []
    if write_drafts:
        for number, cluster in enumerate(proposals, start=1):
            note(f"drafting {number}/{len(proposals)} for {','.join(cluster.ticket_ids)}")
            body = bodies.get(cluster.nearest_page) if cluster.nearest_page else None
            drafts.append(draft_page(client, cluster, body))

    return Result(
        classifications=classifications,
        clusters=clusters,
        drafts=drafts,
        report=summarise(
            classifications,
            pages_proposed=len(proposals),
            tickets_behind_proposals=sum(c.size for c in proposals),
        ),
        entail_calls=entail_calls,
        entail_unparsed=unparsed,
        pair_calls=pair_calls,
        scores=scores,
    )


def load_all() -> tuple[list[Page], list[Ticket]]:
    """The default corpus and ticket batch."""
    return load_pages(), load_tickets()


def open_cache(path: Path) -> EmbeddingCache:
    return EmbeddingCache(path, model=EMBED_MODEL)


def _embed(client: Bedrock, text: str, cache: EmbeddingCache | None) -> list[float]:
    if cache is None:
        return client.embed(text)
    hit = cache.get(text)
    if hit is not None:
        return hit
    vector = client.embed(text)
    cache.put(text, vector)
    return vector
