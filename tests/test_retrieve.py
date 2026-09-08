"""Ranking, deduplication, and the passage text that once went missing."""

from __future__ import annotations

import pytest

from guide_gap.retrieve import Index, cosine


def build() -> Index:
    """Two pages, three sections. Page A's sections point in different directions."""
    return Index(
        page_ids=["a", "a", "b"],
        titles=["A / one", "A / two", "B / one"],
        vectors=[[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]],
        texts=["a-one text", "a-two text", "b-one text"],
    )


def test_search_returns_one_entry_per_page_best_section_first():
    results = build().search([1.0, 0.0], k=3)
    assert [m.page_id for m in results] == ["a", "b"]
    # Page A's *best* section, not whichever one was indexed first.
    assert results[0].text == "a-one text"


def test_search_passages_keeps_several_sections_of_one_page():
    """The correction that fixed four of eight errors in the first entail run.

    Page-deduped candidates mean only a page's best-scoring section is ever
    examined, so a ticket answered by a *different* section of that page is
    reported as an unwritten page.
    """
    results = build().search_passages([1.0, 0.0], k=3)
    assert [m.page_id for m in results] == ["a", "b", "a"]
    assert results[2].text == "a-two text"


def test_matches_carry_their_passage_text():
    """The bug that made every ticket a content gap, with no error anywhere.

    `build_index` once omitted `texts=`, so every passage handed to the coverage
    check was the empty string. The model answered "no, this does not answer the
    question" -- correctly -- and the run looked like a finding.
    """
    for match in build().search_passages([1.0, 0.0], k=3):
        assert match.text


def test_scores_are_cosine_not_dot_product():
    index = Index(["a"], ["A"], [[3.0, 4.0]], ["t"])
    # Unnormalised, [3,4]·[1,0] would be 3. Cosine is 0.6.
    assert index.search([1.0, 0.0])[0].score == pytest.approx(0.6, abs=1e-6)


def test_a_zero_vector_is_similar_to_nothing():
    index = Index(["a"], ["A"], [[0.0, 0.0]], ["t"])
    # Not divided by an epsilon, which would invent a direction and let an empty
    # document rank against real ones.
    assert index.search([1.0, 0.0])[0].score == pytest.approx(0.0)


def test_empty_index_returns_nothing_rather_than_raising():
    index = Index([], [], [])
    assert index.search([1.0, 0.0]) == []
    assert index.search_passages([1.0, 0.0]) == []


def test_pages_counts_distinct_pages_not_rows():
    index = build()
    assert len(index) == 3
    assert index.pages == 2


def test_mismatched_input_lengths_raise():
    with pytest.raises(ValueError):
        Index(["a", "b"], ["A"], [[1.0]])
    with pytest.raises(ValueError):
        Index(["a"], ["A"], [[1.0]], texts=["x", "y"])


def test_cosine_is_symmetric_and_bounded():
    assert cosine([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)
    assert cosine([1.0, 2.0], [2.0, 4.0]) == pytest.approx(1.0, abs=1e-6)
    assert cosine([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)
