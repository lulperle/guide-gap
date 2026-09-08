"""Group the content-gap tickets into candidate guide topics.

Single-link agglomerative clustering over a set of confirmed pairs. Single-link is
right for questions, which chain -- "how do I add an account", "how do I add a
second account", "account creation fails" -- where complete-link would split one
topic into three pages.

The pairing decision arrived at the same wall the coverage decision did. A cosine
threshold was tried first and measured: on this batch the pairs that belong
together score 0.156-0.346 and the pairs that do not score up to 0.311. They
overlap, so no threshold separates them; the best available cut has a margin of
0.013, which is not a threshold, it is a coincidence that happens to fit these
twenty-two tickets.

So similarity is demoted to shortlisting here too. It proposes pairs above a low
floor and a yes/no call confirms each one. That keeps the *result* explainable --
the output is still a list of confirmed pairs, and "ticket 7 is with ticket 12
because these two were judged the same question" is an answer a documentation owner
can argue with, which was the original reason for not letting a model cluster
freely.

The honest limit: pair confirmation is O(pairs), and pairs are quadratic in batch
size. The floor keeps it small here (7 tickets, 6 pairs above 0.25) and would keep
it manageable at a few hundred, but a batch of thousands needs blocking or a
cheaper first pass. That is not solved here and is not pretended to be.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .retrieve import cosine

# Below this, two tickets are not even worth asking about. A shortlist floor, not
# a decision boundary -- set under the lowest-scoring pair that genuinely belongs
# together (0.156 here would need 0.15, but that admits nearly every pair, so this
# sits at the point where the cost stays sane and the loss is recorded rather than
# hidden: pairs below it are never confirmed, and t15-t16 at 0.156 is one of them).
PAIR_FLOOR = 0.25

# Kept for the measured comparison in evals/, not as the recommended path. The
# value is the best cut available on the labelled batch, which is to say it is
# fitted to it.
SAME_TOPIC = 0.32

# A page is worth writing when more than one person asked. A single ticket is a
# conversation, not documentation debt -- and writing a page per one-off is how a
# guide becomes unsearchable, which causes the findability failures this tool is
# supposed to reduce.
MIN_CLUSTER_SIZE = 2


@dataclass
class Cluster:
    """A group of tickets asking the same thing, ranked by how many asked."""

    ticket_ids: list[str]
    texts: list[str]
    nearest_page: str | None

    @property
    def size(self) -> int:
        return len(self.ticket_ids)


def candidate_pairs(
    vectors: list[list[float]], *, floor: float = PAIR_FLOOR
) -> list[tuple[int, int, float]]:
    """Index pairs worth asking about, most similar first.

    Ordered by similarity so that when a caller caps the number of confirmations,
    it spends them on the pairs most likely to be real rather than on whichever
    pair happened to come first in the batch.
    """
    pairs = [
        (i, j, cosine(vectors[i], vectors[j]))
        for i in range(len(vectors))
        for j in range(i + 1, len(vectors))
    ]
    return sorted(
        (p for p in pairs if p[2] >= floor), key=lambda p: (-p[2], p[0], p[1])
    )


def cluster_tickets(
    ticket_ids: list[str],
    texts: list[str],
    vectors: list[list[float]],
    nearest_pages: list[str | None],
    *,
    threshold: float = SAME_TOPIC,
    confirm: Callable[[str, str], bool] | None = None,
    floor: float = PAIR_FLOOR,
    max_confirmations: int | None = None,
) -> list[Cluster]:
    """Group tickets into topics, largest cluster first.

    Args:
        threshold: Used only when `confirm` is None -- the measured-and-rejected
            similarity-threshold path, kept so the eval can compare against it.
        confirm: Called with two ticket texts, returns whether they are the same
            question. When given, similarity only shortlists.
        floor: Similarity below which a pair is never proposed.
        max_confirmations: Hard cap on calls, for when a batch is larger than the
            budget. Pairs are checked most-similar-first, so a cap degrades by
            dropping the least likely pairs rather than by dropping whatever was
            at the end of the list.
    """
    n = len(ticket_ids)
    if not (n == len(texts) == len(vectors) == len(nearest_pages)):
        raise ValueError("all inputs must be the same length")

    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    if confirm is None:
        for i in range(n):
            for j in range(i + 1, n):
                if cosine(vectors[i], vectors[j]) >= threshold:
                    parent[find(i)] = find(j)
    else:
        checked = 0
        for i, j, _ in candidate_pairs(vectors, floor=floor):
            if max_confirmations is not None and checked >= max_confirmations:
                break
            # Already in the same cluster by transitivity, so the call cannot
            # change the outcome. Skipping it is free and cuts the calls on a
            # batch with one large topic, which is the common shape.
            if find(i) == find(j):
                continue
            checked += 1
            if confirm(texts[i], texts[j]):
                parent[find(i)] = find(j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    clusters = [
        Cluster(
            ticket_ids=[ticket_ids[i] for i in members],
            texts=[texts[i] for i in members],
            # The adjacent page most of the cluster pointed at, which is where the
            # new content should probably live. Ties broken by first occurrence so
            # the output is stable across runs.
            nearest_page=_majority([nearest_pages[i] for i in members]),
        )
        for members in groups.values()
    ]
    # Sorted by volume, then by first ticket id so equal-sized clusters do not
    # swap places between runs and turn a diff into noise.
    return sorted(clusters, key=lambda c: (-c.size, c.ticket_ids[0]))


def worth_writing(clusters: list[Cluster], *, minimum: int = MIN_CLUSTER_SIZE) -> list[Cluster]:
    """The clusters that justify a page."""
    return [c for c in clusters if c.size >= minimum]


def _majority(values: list[str | None]) -> str | None:
    present = [v for v in values if v]
    if not present:
        return None
    return max(present, key=lambda v: (present.count(v), -present.index(v)))
