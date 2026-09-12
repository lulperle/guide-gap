"""Drafting: the shape of the output, and what goes into the prompt."""

from __future__ import annotations

from conftest import FakeClient

from guide_gap.cluster import Cluster
from guide_gap.draft import NO_PAGE, Draft, draft_page

WELL_FORMED = """# タグの名前は何にすればいい？

## どんなときに読むページか

- 費用を課ごとに按分したい

## 手順

1. タグを使って集計されます。

## 確認が必要な点

- 命名規則があるかは資料から確認できませんでした。
- タグ付けの操作手順も記載がありません。
"""


def cluster(size: int = 2) -> Cluster:
    return Cluster(
        ticket_ids=[f"t{i}" for i in range(size)],
        texts=[f"問い合わせ{i}" for i in range(size)],
        nearest_page="cost-report",
    )


def test_open_questions_are_extracted_as_a_countable_list():
    """A batch of drafts averaging zero open questions is not a batch of confident
    drafts, it is a prompt that stopped working."""
    draft = Draft(["t1"], "t", WELL_FORMED, "cost-report")
    assert len(draft.open_questions) == 2
    assert draft.open_questions[0].startswith("命名規則")


def test_an_open_question_wrapped_onto_a_second_line_is_kept_whole():
    """Regression. The extractor kept only lines that *begin* with a bullet, and the
    drafts wrap at around forty full-width characters, so nearly every real item ran
    onto a second line and arrived cut off mid-clause. It survived because the thing
    this property is used for is the count, and the count was right."""
    wrapped = """# t

## 確認が必要な点

- タグの名前について、決められた命名規則があるのかどうかは、
  既存ガイドには記載がありません。
- 文字数制限も不明です。
"""
    questions = Draft(["t1"], "t", wrapped, None).open_questions
    assert len(questions) == 2
    assert questions[0] == (
        "タグの名前について、決められた命名規則があるのかどうかは、既存ガイドには記載がありません。"
    )
    # Joined without a separator: these are Japanese lines broken for width, and a
    # space between them reads as a typo in the middle of a sentence.
    assert " " not in questions[0]


def test_a_continuation_is_joined_whether_or_not_it_is_indented():
    """Indentation cannot be required, because the model does not reliably supply
    it: one run wrapped these lines with two spaces and the next run wrapped them
    flush left. Lazy continuation is what CommonMark does anyway."""
    flush_left = """# t

## 確認が必要な点

- タグの名前について、決められた命名規則があるのかどうかは、
既存ガイドには記載がありません。
"""
    assert Draft(["t1"], "t", flush_left, None).open_questions == [
        "タグの名前について、決められた命名規則があるのかどうかは、既存ガイドには記載がありません。"
    ]


def test_a_blank_line_closes_the_item_so_following_prose_is_not_absorbed():
    """The boundary that keeps lazy continuation from swallowing the rest of the
    section: a sentence the model did not mark as a question must not become one."""
    body = """# t

## 確認が必要な点

- 命名規則は不明です。

なお、以上は暫定です。
"""
    assert Draft(["t1"], "t", body, None).open_questions == ["命名規則は不明です。"]


def test_a_draft_that_echoed_the_template_title_is_not_well_formed():
    """Regression. The output template read "# タイトル" under an instruction to use
    the headings exactly as given, so the model did exactly that -- emitted
    "# タイトル" and put the real title on the line below. Every heading was present,
    so the draft passed as well formed and displayed as a finished page."""
    echoed = WELL_FORMED.replace("# タグの名前は何にすればいい？", "# タイトル")
    draft = Draft(["t1"], "タイトル", echoed, None)
    assert not draft.well_formed
    # The headings really are all there; the title is the only thing wrong, which is
    # why the old check missed it.
    assert all(h in echoed for h in ("## どんなときに読むページか", "## 手順", "## 確認が必要な点"))


def test_a_missing_title_is_not_well_formed_either():
    assert not Draft(["t1"], "(no title)", "本文だけ", None).well_formed


def test_well_formed_requires_the_uncertainty_heading():
    draft = Draft(["t1"], "t", WELL_FORMED, None)
    assert draft.well_formed

    missing = WELL_FORMED.replace("## 確認が必要な点", "## おわりに")
    # The failure mode that matters: no place to put uncertainty is where invented
    # procedure shows up.
    assert not Draft(["t1"], "t", missing, None).well_formed


def test_a_body_with_no_headings_has_no_open_questions_and_is_not_well_formed():
    draft = Draft(["t1"], "見出しなし", "本文だけがあります", None)
    assert not draft.well_formed
    assert draft.open_questions == []


def test_draft_page_takes_the_title_from_the_h1():
    client = FakeClient([WELL_FORMED])
    draft = draft_page(client, cluster(), "既存ページの本文")
    assert draft.title == "タグの名前は何にすればいい？"
    assert draft.source_page == "cost-report"
    assert draft.ticket_ids == ["t0", "t1"]


def test_a_reply_with_no_h1_is_labelled_rather_than_backfilled():
    """Falling back to the first ticket's text would make a broken draft look like
    a page somebody wrote."""
    draft = draft_page(FakeClient(["見出しがありません"]), cluster(), "本文")
    assert draft.title == "(no title)"
    assert not draft.well_formed


def test_the_adjacent_page_body_is_the_only_grounding_in_the_prompt():
    client = FakeClient([WELL_FORMED])
    draft_page(client, cluster(), "既存ページの本文はこれです")
    _, prompt, _ = client.prompts[0]
    assert "既存ページの本文はこれです" in prompt
    assert "問い合わせ0" in prompt


def test_a_cluster_with_no_adjacent_page_says_so_instead_of_leaving_it_blank():
    client = FakeClient([WELL_FORMED])
    draft_page(client, Cluster(["t1"], ["問い合わせ"], None), None)
    _, prompt, _ = client.prompts[0]
    assert NO_PAGE in prompt


def test_only_the_first_few_tickets_go_into_the_prompt():
    """Thirty paraphrases of one question add no information after the sixth."""
    client = FakeClient([WELL_FORMED])
    draft = draft_page(client, cluster(size=30), "本文", max_tickets=6)
    _, prompt, _ = client.prompts[0]
    assert "問い合わせ5" in prompt
    assert "問い合わせ6" not in prompt
    # The full cluster size is still what justified the page, and is still reported.
    assert "30件" in prompt
    assert len(draft.ticket_ids) == 30
