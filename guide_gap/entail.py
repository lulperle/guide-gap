"""Does this passage actually answer this question?

This module exists because of a measurement, not a preference. Cosine similarity
between a two-sentence question and a section of a guide page ranks the right page
first most of the time, but the *scores* of answered and unanswered tickets
overlap. Measured on the labelled batch, best-section score per ticket:

    the guide answers it        0.241 - 0.595   (t10 at the bottom)
    the guide does not          0.205 - 0.370   (t14 at the top)

So three answered tickets sit inside the unanswered range. Any cut above 0.370
misclassifies t04, t07 and t10; any cut below 0.241 misclassifies t11, t14 and t17.
Tuning the threshold is not the fix -- it only moves which cases are wrong.

Similarity measures topical closeness. "Is the answer in here" is entailment, and
those are different questions. A page about billing is topically close to every
billing question ever asked, including the ones it says nothing about.

So the similarity search is demoted to what it is good at -- shortlisting -- and a
single cheap yes/no call decides coverage. The call is deliberately minimal: one
passage, one question, a handful of output tokens, no reasoning text to parse.

What this is *not* is a judge scoring the eval. The labels in `fixtures/` were
written by hand before any of this ran, and the eval compares this module's output
against them by exact match. A model doing product work and a model grading its own
homework are different things, and only the second one launders a bad result.
"""

from __future__ import annotations

from dataclasses import dataclass

from .bedrock import Bedrock

SYSTEM = """与えられた「問い合わせ」に対して、「ガイド抜粋」を読んだ利用者が
「次に何をすればよいか」を判断できるかどうかを判定してください。

「はい」と答える条件:
- 問い合わせの内容に対応する手順・条件・数値・依頼先が抜粋に書かれている。
- 「できない」「対応していない」ことと、その場合にどうするかが書かれている。
  希望どおりの方法がない場合でも、次の行動が決まるなら「はい」です。
- 抜粋の言葉が問い合わせの言葉と違っていても、同じものを指しているなら「はい」。
  利用者が別の呼び方をしているだけです。

「いいえ」と答える条件:
- 抜粋は同じ話題を扱っているが、聞かれていることには触れていない。
- 制度や前提の説明はあるが、次の行動が決まらない。

話題が近いこと、関連する制度が書かれていることは「はい」の理由になりません。
問い合わせの問いそのものに対応する記述が必要です。抜粋に書かれていないことを
あなたの知識で補って「はい」にしてはいけません。

出力は「はい」または「いいえ」のみ。理由や前置きを書かないでください。"""

PROMPT = """## 問い合わせ

{ticket}

## ガイド抜粋

{passage}
"""

YES = "はい"
NO = "いいえ"

# Room for the word, plus room for the model to get there.
#
# This was 8. Claude Sonnet 5 emits a `reasoningContent` block before its text and
# that block is charged against maxTokens, so at 8 the entire budget went to
# reasoning, the response came back with no text block at all, and the empty string
# parsed as 「いいえ」. Every truncated call became a confident negative -- across
# three repeats, stably, which is what made it look like a real judgement.
#
# 64 was still short: 8 of 64 calls in a full pass still truncated. The observed
# reasoning length for these two-line questions is around 160 output tokens, so
# this is set well clear of it rather than close to it.
#
# Turning reasoning off is possible -- `additionalModelRequestFields={"thinking":
# {"type": "disabled"}}` returns 「はい」/「いいえ」 in 6 tokens -- and it changes the
# answers. On the pair that started this investigation, reasoning on gives the
# correct verdict and reasoning off gives the wrong one. So the cheap path is
# available and measurably worse; it is not taken.
#
# `Usage.truncated` counts `stopReason == "max_tokens"` so this cannot recur
# silently.
MAX_TOKENS = 256


@dataclass(frozen=True)
class Verdict:
    """One coverage decision, with the raw reply kept."""

    answered: bool
    raw: str
    page_id: str

    @property
    def parsed(self) -> bool:
        """Whether the reply was actually one of the two expected words.

        Reported separately from `answered`. An unparsed reply is counted as "not
        answered", which is the conservative direction -- it proposes writing a page
        that may already exist, and a redundant draft gets rejected in review,
        whereas a wrongly-dismissed gap is invisible and stays open. But conflating
        the two would hide a broken prompt behind a plausible-looking number.
        """
        return self.raw.strip().startswith((YES, NO))


def answers(client: Bedrock, ticket: str, passage: str, page_id: str = "") -> Verdict:
    """Ask whether one passage answers one ticket.

    Raises on an empty passage rather than asking. An empty passage gets a
    perfectly reasonable 「いいえ」, so the batch completes, reports every ticket as
    a content gap, and records no errors -- a wrong answer that looks like a
    finding. This guard exists because that already happened once, when the index
    was built without passing the passage text through.
    """
    if not passage.strip():
        raise ValueError(f"empty passage for page {page_id!r}: nothing to check against")

    reply = client.draft(
        SYSTEM,
        PROMPT.format(ticket=ticket.strip(), passage=passage.strip()),
        max_tokens=MAX_TOKENS,
    )
    stripped = reply.strip()
    # Checked in this order because 「いいえ」 does not contain 「はい」 but a reply
    # like 「はい、いいえのどちらとも…」 would match both; the negative reading is the
    # safe one.
    if stripped.startswith(NO):
        return Verdict(False, reply, page_id)
    return Verdict(stripped.startswith(YES), reply, page_id)


SAME_QUESTION_SYSTEM = """2件の問い合わせが与えられます。これから新しいガイドページを
1つ作るとして、この2件が**同じページの中**に収まるかどうかを判定してください。

「はい」と答える条件:
- 利用者が置かれている場面が同じ。作業や手続きが同じで、そのどの段階を
  聞いているかが違うだけ。
- 節を分ければ1ページに両方書ける。聞かれている具体的な内容が違っていても、
  同じ場面についての疑問なら「はい」です。

「いいえ」と答える条件:
- 場面が別。片方を読みに来た利用者にとって、もう片方は関係のない内容になる。
- 同じ費用や同じ権限の話でも、発生する場面が違うなら「いいえ」。

判断の基準は「この2件を1ページにまとめて、両方の利用者が得をするか」です。

出力は「はい」または「いいえ」のみ。"""

SAME_QUESTION_PROMPT = """## 問い合わせA

{first}

## 問い合わせB

{second}
"""


def same_question(client: Bedrock, first: str, second: str) -> Verdict:
    """Whether two tickets belong on one page.

    Phrased as "would one page serve both" rather than "are these similar",
    because similarity is what already failed here. The question a documentation
    owner actually faces is whether merging helps the reader, and two tickets can
    be about the same service, the same page even, and still need separate
    sections.

    Returns a Verdict rather than a bool so an unparsed reply stays visible. The
    bool version of this function returned False for both "no" and "the call came
    back empty", which is how a truncation bug survived a run that looked stable
    across three repeats.
    """
    reply = client.draft(
        SAME_QUESTION_SYSTEM,
        SAME_QUESTION_PROMPT.format(first=first.strip(), second=second.strip()),
        max_tokens=MAX_TOKENS,
    )
    stripped = reply.strip()
    if stripped.startswith(NO):
        return Verdict(False, reply, "")
    return Verdict(stripped.startswith(YES), reply, "")


def first_covering(
    client: Bedrock,
    ticket: str,
    matches,
    *,
    limit: int = 3,
) -> tuple[Verdict | None, int]:
    """Check candidates best-first and stop at the first one that answers.

    Returns the deciding verdict (or the last negative one) and how many calls it
    took. Checking more than the top match matters more than it looks: on this
    batch the right passage was ranked second or lower for several tickets, because
    a question phrased as a complaint ("it came back rejected") is closer to the
    incident page than to the page stating the rule it broke.

    Note what the loop stops on: the *first* affirmative. That biases towards
    finding coverage, so a wrongly-affirmative passage ends the search and hides a
    real gap. It is the direction that costs the most, and the reason the prompt
    spends more words forbidding topical matches than allowing paraphrases.
    """
    last: Verdict | None = None
    calls = 0
    for match in matches[:limit]:
        verdict = answers(client, ticket, match.text, match.page_id)
        calls += 1
        last = verdict
        if verdict.answered:
            return verdict, calls
    return last, calls
