"""Similarity search over the guide corpus. Pure arithmetic, no network.

Kept separate from the Bedrock client so that everything downstream can be tested
with hand-written vectors. The interesting failures in a retrieval pipeline are in
the ranking and the thresholds, not in the HTTP call, and those are exactly the
parts a mocked client would hide.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Match:
    """One guide page, its best passage, and how close that passage was."""

    page_id: str
    title: str
    score: float
    # The passage itself, carried along rather than looked up again later. The
    # decision about whether the guide answers a ticket is made against this text,
    # so keeping it attached to the score means the evidence and the number cannot
    # drift apart.
    text: str = ""


class Index:
    """Searchable embeddings, one row per unit indexed.

    A row is a *section* of a page, not a whole page, and a page's score is the
    best score among its sections. That is not a refinement; it is the difference
    between this working and not working, and it was measured -- see the numbers in
    the README.

    Embedding whole pages puts every ticket in the 0.17-0.61 band regardless of
    whether the page answered it, because a 400-word procedure page and a
    two-sentence question differ mostly in length and register, and that dominates
    the cosine. The answer to "I changed phones" is one section of the MFA page;
    averaged against the other five sections it disappears. Max-over-sections asks
    the question that actually matters: is there a *passage* here that answers this.
    """

    def __init__(
        self,
        page_ids: list[str],
        titles: list[str],
        vectors: list[list[float]],
        texts: list[str] | None = None,
    ):
        if not (len(page_ids) == len(titles) == len(vectors)):
            raise ValueError("page_ids, titles and vectors must be the same length")
        if texts is not None and len(texts) != len(page_ids):
            raise ValueError("texts must be the same length as page_ids")
        self.page_ids = page_ids
        self.titles = titles
        self.texts = texts if texts is not None else [""] * len(page_ids)
        matrix = np.asarray(vectors, dtype=np.float32)
        # An empty list arrives one-dimensional, and normalising along axis 1 then
        # raises. The search methods already return nothing for an empty index, but
        # construction happened first, so "no guide pages yet" crashed instead of
        # reporting every ticket as a content gap -- which is the correct answer for
        # a corpus that does not exist.
        if matrix.size == 0:
            matrix = matrix.reshape(0, 0)
        # Normalised once at construction. Cosine similarity then reduces to a
        # dot product, which keeps search a single matrix multiply and removes the
        # chance of normalising in one place and forgetting in another.
        self._matrix = _normalise(matrix)

    def __len__(self) -> int:
        return len(self.page_ids)

    @property
    def pages(self) -> int:
        """Distinct pages indexed, as opposed to rows."""
        return len(set(self.page_ids))

    def _scores(self, vector: list[float]) -> np.ndarray:
        """Cosine similarity of one query against every row."""
        query = _normalise(np.asarray([vector], dtype=np.float32))[0]
        return self._matrix @ query

    def search_passages(self, vector: list[float], k: int = 5) -> list[Match]:
        """The k most similar passages, best first, with no per-page limit.

        This is the shortlist for the coverage check, and keeping it separate from
        `search` is a correction, not a convenience. Deduplicating to one entry per
        page means only a page's single best-scoring section is ever examined, so
        when the answer lives in a *different* section of that page it is never
        looked at and the ticket is reported as an unwritten page. Measured: that
        one mistake cost four of the eight errors in the first entailment run.

        The cost is real -- several sections of the same page can fill the
        shortlist, crowding out other pages -- so `k` here is a dial with a
        measured effect, not a free improvement.
        """
        if len(self) == 0:
            return []
        scores = self._scores(vector)
        order = np.argsort(scores)[::-1][:k]
        return [
            Match(self.page_ids[i], self.titles[i], float(scores[i]), self.texts[i])
            for i in order
        ]

    def search(self, vector: list[float], k: int = 3) -> list[Match]:
        """The k most similar pages, best first, one entry per page.

        Page-level, for reporting: a findability report that lists the same page
        three times tells a reviewer nothing about the alternatives.
        """
        if len(self) == 0:
            return []
        scores = self._scores(vector)
        # argsort ascending then reversed, rather than argpartition: the corpus is
        # a guide rather than a web index, and a stable full sort keeps ties in a
        # predictable order so the tests are not flaky.
        order = np.argsort(scores)[::-1]

        results: list[Match] = []
        seen: set[str] = set()
        for i in order:
            page_id = self.page_ids[i]
            # First occurrence wins, and because the order is descending that is
            # the page's best section. Returning the same page twice would let one
            # well-written page crowd out the alternatives a reviewer needs to see.
            if page_id in seen:
                continue
            seen.add(page_id)
            results.append(
                Match(page_id, self.titles[i], float(scores[i]), self.texts[i])
            )
            if len(results) == k:
                break
        return results


def _normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A zero vector has no direction, so it cannot be similar to anything. Left
    # as zeros rather than divided by an epsilon, which would invent a direction
    # and quietly rank an empty document against real ones.
    norms[norms == 0] = 1.0
    return matrix / norms


def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    pair = _normalise(np.asarray([a, b], dtype=np.float32))
    return float(pair[0] @ pair[1])
