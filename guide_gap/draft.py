"""Draft a guide page from a cluster of tickets.

What the model is asked to do here is narrow on purpose. It writes the *page
skeleton* -- the title people would search for, the question in the user's own
words, the steps that are already evidenced by the adjacent guide page -- and it
is required to list what it does not know as open questions.

That last part is the whole reason this stage is safe to automate. A drafting
prompt with no escape hatch has exactly one way to satisfy it: invent the
procedure. Given "write the steps" and no source, a model will produce plausible,
confident, wrong instructions for granting a permission in a government cloud
environment, and the reviewer's job silently changes from editing to fact-
checking -- which is slower than writing the page from scratch, so the tool makes
the work worse while appearing to help.

So the deal is: draft what the corpus supports, and name the gaps. A page arriving
with three open questions attached is useful. The same page with those three
answered from nowhere is a liability.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .bedrock import Bedrock
from .cluster import Cluster

SYSTEM = """あなたは行政向けクラウドサービスの利用ガイドを書く担当者です。

問い合わせの実例と、既存ガイドの関連ページが与えられます。新しいガイドページの
下書きを作成してください。

厳守事項:
- 手順は、与えられた既存ページに書かれている内容だけから書くこと。
- 与えられた資料から確認できないことは、絶対に書かないでください。代わりに
  「確認が必要な点」に箇条書きで挙げること。推測した手順は、丁寧な言い方をしても
  推測です。
- 画面名、メニュー名、コマンド、URL、権限名は、資料に出てきたものだけを使うこと。
  それらしい名前を作ってはいけません。
- タイトルは、利用者が検索するときに打つ言葉にすること。制度上の正式名称ではなく、
  問い合わせに実際に出てきた言葉を優先すること。

出力形式:

1行目は「# 」で始め、そのページのタイトルを書くこと。利用者が検索するときに打つ
言葉にすること。「タイトル」という語をそのまま書かないでください。

そのあとに、次の3つの見出しをこの表記のまま、この順で使うこと:

## どんなときに読むページか

## 手順

## 確認が必要な点
"""

PROMPT = """## 同じ内容の問い合わせ（{count}件）

{tickets}

## 既存ガイドの関連ページ

{page}
"""

NO_PAGE = "（関連するページは見つかりませんでした）"

# The headings the system prompt demands. Checked rather than trusted: a draft
# missing "確認が必要な点" is the exact failure mode that matters, because it means
# the model had nothing to say about its own uncertainty, and that is when invented
# procedure shows up.
REQUIRED_HEADINGS = ("## どんなときに読むページか", "## 手順", "## 確認が必要な点")

# Titles that mean the model copied the output template instead of filling it in.
#
# This is here because it happened. The template line used to read "# タイトル" under
# an instruction to use the headings exactly as given, so the model did precisely
# that -- emitted "# タイトル" and put the real title on the line below. The prompt is
# fixed, but a placeholder title is the specific shape prompt drift takes here, and
# the point of `well_formed` is to catch drift rather than to trust that it stopped.
PLACEHOLDER_TITLES = ("タイトル", "(no title)")


@dataclass
class Draft:
    """A proposed guide page and the tickets that justify it."""

    ticket_ids: list[str]
    title: str
    body: str
    source_page: str | None

    @property
    def open_questions(self) -> list[str]:
        """The bullets under 確認が必要な点, each joined back into one sentence.

        Extracted rather than left in prose because the count is a reviewable
        number: a batch of drafts averaging zero open questions is not a batch of
        confident drafts, it is a prompt that stopped working.

        The continuation handling is a fix, not a refinement. This used to keep only
        the lines that *start* with a bullet, and the drafts wrap at around forty
        full-width characters, so nearly every item ran onto a second line and
        arrived cut off mid-clause -- 「命名規則があるのかどうかは、」 and nothing after
        it. The bug survived because the thing this property is used for is the
        count, and the count was right.
        """
        items: list[str] = []
        open_item = False
        for line in _section(self.body, "## 確認が必要な点").splitlines():
            stripped = line.strip()
            if not stripped:
                # A blank line closes the item. Without this, prose following the
                # list gets appended to the last question.
                open_item = False
                continue
            if stripped.startswith(("-", "*", "・")):
                items.append(stripped.lstrip("-*・ 　").strip())
                open_item = True
            elif open_item:
                # A continuation, indented or not. Indentation is not required here
                # because the model does not reliably supply it: one run wrapped
                # these lines with two spaces and the next run wrapped them flush
                # left. Lazy continuation is also what CommonMark does, so the
                # looser rule is the more standard one as well. Joined without a
                # separator -- these are Japanese lines broken for width, and a
                # space between them reads as a typo mid-sentence.
                items[-1] += stripped
        return items

    @property
    def well_formed(self) -> bool:
        """Whether the draft has the shape the prompt asked for.

        Includes the title, not just the headings. A draft can carry all three
        required headings and still be the output template echoed back -- see
        `PLACEHOLDER_TITLES` -- and that reads as a finished page everywhere it is
        displayed.
        """
        if self.title.strip() in PLACEHOLDER_TITLES:
            return False
        return all(h in self.body for h in REQUIRED_HEADINGS)


def draft_page(
    client: Bedrock,
    cluster: Cluster,
    page_body: str | None,
    *,
    max_tickets: int = 6,
) -> Draft:
    """Draft one page for one cluster.

    Only the first `max_tickets` tickets go into the prompt. A cluster of thirty
    paraphrases of one question adds no information after the sixth and costs
    tokens on every one, so the cap is a straight saving with no effect on the
    output. The full cluster size is still what justified the page and is still
    reported.
    """
    tickets = "\n".join(f"- {t.strip()}" for t in cluster.texts[:max_tickets])
    prompt = PROMPT.format(
        count=cluster.size,
        tickets=tickets,
        page=page_body.strip() if page_body else NO_PAGE,
    )
    body = client.draft(SYSTEM, prompt)
    return Draft(
        ticket_ids=list(cluster.ticket_ids),
        title=_title(body),
        body=body,
        source_page=cluster.nearest_page,
    )


def _title(body: str) -> str:
    """The H1, or a placeholder.

    An empty title is left visible instead of falling back to the first ticket's
    text. A draft that failed to produce a heading should look broken in the
    output, not look like a page somebody wrote.
    """
    match = re.search(r"^#\s+(.+)$", body, re.MULTILINE)
    return match.group(1).strip() if match else "(no title)"


def _section(body: str, heading: str) -> str:
    """The text under one heading, up to the next heading of any level."""
    lines = body.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == heading)
    except StopIteration:
        return ""
    collected = []
    for line in lines[start + 1 :]:
        if line.startswith("#"):
            break
        collected.append(line)
    return "\n".join(collected)
