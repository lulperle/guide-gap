"""Reply parsing. This is where the worst bug in the project lived."""

from __future__ import annotations

import pytest
from conftest import FakeClient

from guide_gap.entail import MAX_TOKENS, answers, first_covering, same_question
from guide_gap.retrieve import Match

PASSAGE = "端末を交換した場合はサービスデスクへ連絡してください。"


def test_yes_and_no_are_parsed():
    assert answers(FakeClient(["はい"]), "質問", PASSAGE).answered is True
    assert answers(FakeClient(["いいえ"]), "質問", PASSAGE).answered is False


def test_an_empty_reply_counts_as_not_answered_but_is_flagged_unparsed():
    """The truncation bug, pinned.

    Claude Sonnet 5 emits reasoning before its text and that reasoning is charged
    against maxTokens. At a small cap the response came back with no text block at
    all, the empty string parsed as 「いいえ」, and every truncated call became a
    confident negative -- stably, across three repeats, which is what made it look
    like real judgement. `answered` stays conservative; `parsed` is what makes it
    visible.
    """
    verdict = answers(FakeClient([""]), "質問", PASSAGE)
    assert verdict.answered is False
    assert verdict.parsed is False


def test_a_chatty_reply_is_flagged_unparsed_too():
    verdict = answers(FakeClient(["この抜粋では判断できません"]), "質問", PASSAGE)
    assert verdict.answered is False
    assert verdict.parsed is False


def test_an_ambiguous_reply_reads_as_negative():
    """「はい」 appears in it, but so does 「いいえ」 at the front. Safe reading wins."""
    verdict = answers(FakeClient(["いいえ、はいとは言えません"]), "質問", PASSAGE)
    assert verdict.answered is False
    assert verdict.parsed is True


def test_an_empty_passage_raises_instead_of_being_asked_about():
    """An empty passage gets a perfectly reasonable 「いいえ」.

    So the batch completes, reports every ticket as a content gap, and records no
    errors. That already happened once, when the index was built without passing
    the passage text through.
    """
    client = FakeClient(["はい"])
    with pytest.raises(ValueError):
        answers(client, "質問", "   ", "mfa")
    assert client.calls == 0


def test_the_token_budget_is_well_clear_of_the_observed_reasoning_length():
    # Observed reasoning for these two-line questions is around 160 output tokens.
    # 8 truncated every call; 64 still truncated 8 of 64 in a full pass.
    assert MAX_TOKENS >= 256


def test_the_page_id_travels_with_the_verdict():
    verdict = answers(FakeClient(["はい"]), "質問", PASSAGE, "mfa-setup")
    assert verdict.page_id == "mfa-setup"


def test_same_question_returns_a_verdict_so_unparsed_stays_visible():
    """It used to return a bool, so "no" and "the call came back empty" were the
    same value. That is how the truncation bug survived a stable-looking run."""
    assert same_question(FakeClient(["はい"]), "A", "B").answered is True
    empty = same_question(FakeClient([""]), "A", "B")
    assert empty.answered is False
    assert empty.parsed is False


def shortlist() -> list[Match]:
    return [
        Match("incident", "障害", 0.55, "障害発生時の連絡先です。"),
        Match("account", "払出", 0.51, "払い出しは5営業日です。"),
        Match("cost", "費用", 0.44, "費用は月次で確定します。"),
    ]


def test_first_covering_stops_at_the_first_affirmative():
    client = FakeClient(["いいえ", "はい", "はい"])
    verdict, calls = first_covering(client, "質問", shortlist())
    assert calls == 2
    assert verdict.answered is True
    assert verdict.page_id == "account"


def test_first_covering_returns_the_last_negative_when_nothing_answers():
    client = FakeClient(["いいえ", "いいえ", "いいえ"])
    verdict, calls = first_covering(client, "質問", shortlist())
    assert calls == 3
    assert verdict.answered is False
    assert verdict.page_id == "cost"


def test_first_covering_respects_the_call_budget():
    client = FakeClient(["いいえ", "いいえ", "はい"])
    verdict, calls = first_covering(client, "質問", shortlist(), limit=2)
    assert calls == 2
    assert verdict.answered is False


def test_first_covering_on_an_empty_shortlist_makes_no_calls():
    client = FakeClient(["はい"])
    verdict, calls = first_covering(client, "質問", [])
    assert (verdict, calls) == (None, 0)
    assert client.calls == 0
