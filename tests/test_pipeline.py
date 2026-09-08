"""End to end with a fake client. The wiring is where both real bugs lived.

Neither of them was in a function -- every unit was correct. `build_index` forgot
to pass the passage text, and the coverage check was handed page-deduplicated
candidates. Both produced a plausible report and no error, which is exactly the
class of bug a unit-only suite cannot see.
"""

from __future__ import annotations

from conftest import FakeClient

from guide_gap.corpus import Page, Ticket
from guide_gap.coverage import Cause
from guide_gap.pipeline import analyse, build_index

PAGES = [
    Page(
        "mfa-setup",
        "多要素認証の設定",
        ["MFA", "ワンタイムパスワード"],
        "## 初回の設定\n認証アプリでQRコードを読み取ります。\n"
        "\n## 端末を交換したとき\n古い端末が使えないときはサービスデスクへ連絡します。\n",
    ),
    Page(
        "cost-report",
        "費用の確認",
        ["請求", "按分"],
        "## 明細の出し方\nポータルからCSVを出力します。\n",
    ),
]

TICKETS = [
    Ticket("t1", "スマホを買い替えたのでMFAを移したい", "findability_gap", False, "mfa-setup"),
    Ticket("t2", "権限を付与してほしい", "not_deflectable", True),
]


def test_the_index_carries_every_section_with_its_text():
    index = build_index(FakeClient(), PAGES, None)
    assert len(index) == 3
    assert index.pages == 2
    # The omission that made every ticket a content gap. Asserted on the whole
    # index rather than on one row, because one populated row was never the bug.
    assert all(text.strip() for text in index.texts)


def test_whole_page_indexing_is_still_available_for_the_comparison():
    index = build_index(FakeClient(), PAGES, None, by_section=False)
    assert len(index) == 2
    assert all(text.strip() for text in index.texts)


def test_a_ticket_needing_a_human_costs_no_model_calls():
    """Not an optimisation. The classification is identical either way, so the call
    can only add cost and a chance of a wrong answer."""
    client = FakeClient(["はい"])
    result = analyse(client, PAGES, TICKETS, write_drafts=False, floor=0.0)
    human = next(c for c in result.classifications if c.ticket_id == "t2")
    assert human.cause is Cause.NOT_DEFLECTABLE
    # One ticket was checkable, so at most that one produced calls -- never t2.
    assert all("権限を付与してほしい" not in prompt for _, prompt, _ in client.prompts)


def test_an_affirmative_verdict_becomes_a_findability_gap_with_a_named_page():
    result = analyse(
        FakeClient(["はい"]), PAGES, TICKETS, write_drafts=False, floor=0.0, candidates=1
    )
    item = next(c for c in result.classifications if c.ticket_id == "t1")
    assert item.cause is Cause.FINDABILITY_GAP
    assert item.decided_on in {"mfa-setup", "cost-report"}
    assert result.entail_calls == 1
    assert result.report.per_page_findability


def test_negative_verdicts_produce_content_gaps_and_get_clustered():
    # Two negatives per checkable ticket, then a pair confirmation. Only t1 is
    # checkable here, so one cluster of one and no draft.
    client = FakeClient(["いいえ", "いいえ", "いいえ", "いいえ", "いいえ"])
    result = analyse(client, PAGES, TICKETS, write_drafts=False, floor=0.0)
    item = next(c for c in result.classifications if c.ticket_id == "t1")
    assert item.cause is Cause.CONTENT_GAP
    assert [c.ticket_ids for c in result.clusters] == [["t1"]]
    assert result.report.pages_proposed == 0


def test_unparsed_replies_are_counted_not_swallowed():
    result = analyse(
        FakeClient([""] * 8), PAGES, TICKETS, write_drafts=False, floor=0.0
    )
    assert result.entail_unparsed >= 1


def test_the_threshold_baseline_makes_no_model_calls_at_all():
    """Which is why the deterministic half of this pipeline runs in CI."""
    client = FakeClient()
    result = analyse(
        client, PAGES, TICKETS, write_drafts=False, entail=False, covered=0.9
    )
    assert client.calls == 0
    assert result.entail_calls == 0
    assert result.pair_calls == 0
    assert len(result.classifications) == 2


def test_progress_is_reported_per_ticket():
    """A batch that prints nothing for two minutes gets interrupted by whoever is
    watching it, which is what happened the first time this ran."""
    lines: list[str] = []
    analyse(
        FakeClient(),
        PAGES,
        TICKETS,
        write_drafts=False,
        entail=False,
        progress=lines.append,
    )
    assert any("t1" in line for line in lines)
    assert any("t2" in line for line in lines)
