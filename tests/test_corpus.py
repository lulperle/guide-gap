"""Loading and splitting. Includes the real corpus, on purpose."""

from __future__ import annotations

import pytest

from guide_gap.corpus import Page, load_pages, load_tickets

PAGE = Page(
    page_id="mfa",
    title="多要素認証の設定",
    keywords=["MFA", "ワンタイムパスワード"],
    body=(
        "このページは利用者ポータルにログインする職員向けです。\n"
        "\n"
        "## 初回の設定\n"
        "認証アプリを入れてQRコードを読み取ります。\n"
        "\n"
        "## 端末を交換したとき\n"
        "古い端末が使えない場合はサービスデスクへ連絡します。\n"
    ),
)


def test_sections_split_on_headings_and_keep_the_preamble():
    sections = PAGE.sections()
    assert [s.heading for s in sections] == [
        "（前書き）",
        "初回の設定",
        "端末を交換したとき",
    ]
    # The preamble is the only sentence saying who the page is for. Dropping it is
    # how a page becomes unfindable by the people it was written for.
    assert "職員向け" in sections[0].text


def test_embedding_text_carries_the_page_title_and_keywords():
    section = PAGE.sections()[2]
    # Without this, 「古い端末が…」 is about nothing in particular and the section
    # cannot be retrieved by anyone searching for MFA.
    assert "多要素認証" in section.embedding_text
    assert "ワンタイムパスワード" in section.embedding_text
    assert "サービスデスク" in section.embedding_text


def test_a_page_with_no_headings_is_still_indexed():
    page = Page("faq", "よくある質問", [], "問い合わせ先は下記です。")
    sections = page.sections()
    # One section holding the whole body, rather than zero sections and a page that
    # silently vanishes from the index.
    assert len(sections) == 1
    assert "問い合わせ先" in sections[0].text


def test_missing_front_matter_raises_rather_than_guessing(tmp_path):
    (tmp_path / "broken.md").write_text("# no front matter\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_pages(tmp_path)


def test_no_pages_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pages(tmp_path)


def test_the_real_corpus_loads_and_every_page_has_sections():
    pages = load_pages()
    assert len(pages) >= 5
    for page in pages:
        assert page.title
        assert page.sections()


def test_every_ticket_is_labelled_with_a_cause_the_pipeline_can_produce():
    tickets = load_tickets()
    causes = {"content_gap", "findability_gap", "not_deflectable"}
    assert {t.expect_cause for t in tickets} <= causes
    # All three buckets have to be represented, or the accuracy number is measuring
    # a two-way decision while claiming to measure a three-way one.
    assert {t.expect_cause for t in tickets} == causes
    assert len({t.ticket_id for t in tickets}) == len(tickets)


def test_findability_labels_name_a_page_that_exists():
    """A findability label without a real target page is not checkable by hand.

    And a typo in `expect_page` would make a correct prediction score as wrong,
    which looks exactly like a retrieval regression.
    """
    page_ids = {p.page_id for p in load_pages()}
    for ticket in load_tickets():
        if ticket.expect_cause == "findability_gap":
            assert ticket.expect_page, f"{ticket.ticket_id} has no expect_page"
            assert ticket.expect_page in page_ids, ticket.ticket_id


def test_not_deflectable_tickets_are_marked_as_needing_a_human():
    for ticket in load_tickets():
        if ticket.expect_cause == "not_deflectable":
            assert ticket.needs_human, ticket.ticket_id
