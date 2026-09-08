"""Grouping, and what the confirmation callback is allowed to cost."""

from __future__ import annotations

import pytest

from guide_gap.cluster import Cluster, candidate_pairs, cluster_tickets, worth_writing

# Three tickets in a line: 0 is close to 1, 1 is close to 2, 0 and 2 are far apart.
CHAIN = [[1.0, 0.0], [0.75, 0.66], [0.0, 1.0]]


def group(**kwargs) -> list[Cluster]:
    return cluster_tickets(
        ticket_ids=["t1", "t2", "t3"],
        texts=["一つ目", "二つ目", "三つ目"],
        vectors=CHAIN,
        nearest_pages=["cost", "cost", None],
        **kwargs,
    )


def test_single_link_chains_transitively():
    """Questions chain. Complete-link would split one topic into three pages."""
    clusters = group(confirm=lambda a, b: True)
    assert len(clusters) == 1
    assert clusters[0].ticket_ids == ["t1", "t2", "t3"]


def test_a_refused_pair_stays_separate():
    clusters = group(confirm=lambda a, b: False)
    assert [c.size for c in clusters] == [1, 1, 1]


def test_transitively_merged_pairs_are_not_re_confirmed():
    """The saving that keeps the call count sane on a batch with one big topic."""
    asked: list[tuple[str, str]] = []

    def confirm(a: str, b: str) -> bool:
        asked.append((a, b))
        return True

    group(confirm=confirm, floor=0.0)
    # Three pairs exist above a zero floor; two merges are enough to join three
    # tickets, so the third pair cannot change the outcome and is skipped.
    assert len(asked) == 2


def test_max_confirmations_spends_its_budget_on_the_likeliest_pairs():
    asked: list[tuple[str, str]] = []

    def confirm(a: str, b: str) -> bool:
        asked.append((a, b))
        return True

    clusters = group(confirm=confirm, floor=0.0, max_confirmations=1)
    assert len(asked) == 1
    # The most similar pair, not whichever came first in the batch.
    assert asked[0] == ("一つ目", "二つ目")
    assert [c.size for c in clusters] == [2, 1]


def test_the_threshold_path_still_works_for_the_baseline():
    clusters = group(threshold=0.9)
    assert [c.size for c in clusters] == [1, 1, 1]
    assert len(group(threshold=0.5)) == 1


def test_candidate_pairs_are_ordered_and_floored():
    pairs = candidate_pairs(CHAIN, floor=0.5)
    scores = [p[2] for p in pairs]
    assert scores == sorted(scores, reverse=True)
    # (0, 2) are orthogonal, so the floor drops them.
    assert (0, 2) not in [(i, j) for i, j, _ in pairs]


def test_nearest_page_is_the_majority_of_the_cluster():
    clusters = group(confirm=lambda a, b: True)
    # Two of three pointed at cost; the ticket with no adjacent page does not win.
    assert clusters[0].nearest_page == "cost"


def test_a_cluster_with_no_adjacent_page_reports_none():
    clusters = cluster_tickets(
        ticket_ids=["t1"], texts=["x"], vectors=[[1.0, 0.0]], nearest_pages=[None]
    )
    assert clusters[0].nearest_page is None


def test_clusters_sort_by_volume_then_by_id_for_a_stable_diff():
    clusters = cluster_tickets(
        ticket_ids=["t9", "t1", "t2"],
        texts=["a", "b", "c"],
        vectors=[[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        nearest_pages=[None, None, None],
        confirm=lambda a, b: True,
        floor=0.9,
    )
    assert [c.ticket_ids for c in clusters] == [["t1", "t2"], ["t9"]]


def test_worth_writing_excludes_one_off_questions():
    """A single ticket is a conversation, not documentation debt."""
    clusters = group(confirm=lambda a, b: False)
    assert worth_writing(clusters, minimum=2) == []
    assert len(worth_writing(clusters, minimum=1)) == 3


def test_mismatched_input_lengths_raise():
    with pytest.raises(ValueError):
        cluster_tickets(["t1"], ["a", "b"], [[1.0]], [None])
