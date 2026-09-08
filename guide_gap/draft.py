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

出力形式（この見出しをそのまま使うこと）:

# タイトル

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


@dataclass
class Draft:
    """A proposed guide page and the tickets that justify it."""

    ticket_ids: list[str]
    title: str
    body: str
    source_page: str | None

    @property
    def open_questions(self) -> list[str]:
        """The bullets under 確認が必要な点.

        Extracted rather than left in prose because the count is a reviewable
        number: a batch of drafts averaging zero open questions is not a batch of
        confident drafts, it is a prompt that stopped working.
        """
        section = _section(self.body, "## 確認が必要な点")
        return [
            line.lstrip("-*・ 　").strip()
            for line in section.splitlines()
            if line.strip().startswith(("-", "*", "・"))
        ]

    @property
    def well_formed(self) -> bool:
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
