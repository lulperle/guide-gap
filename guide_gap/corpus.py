"""Load the guide pages and the ticket batch.

The guide is markdown on disk with YAML front matter, which is what a real
documentation repository looks like, so the ingest path is the real one rather
than a fixture format invented for this project.

Tickets carry labels. Every ticket in `fixtures/tickets.yaml` records what the
right answer is -- the cause, and whether it needs a human -- written down before
any model saw it. That is what makes the accuracy number in `evals/` a measurement
rather than an impression.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
GUIDE_DIR = ROOT / "corpus" / "guide"
TICKETS = ROOT / "fixtures" / "tickets.yaml"


@dataclass(frozen=True)
class Section:
    """One `##` section of a page: the unit that actually gets embedded."""

    page_id: str
    page_title: str
    heading: str
    text: str

    @property
    def embedding_text(self) -> str:
        """Section text, with the page's title and keywords prepended.

        The context matters because a section reads as an orphan without it --
        「古い端末が手元にある場合は…」 is about nothing in particular until you know
        the page is about multi-factor authentication. The keywords are also the
        honest place to put the words users actually type, which are rarely the
        words the page is titled with; that mismatch is the findability problem
        this tool measures.
        """
        return f"{self.page_title}\n{self.heading}\n\n{self.text}"


@dataclass(frozen=True)
class Page:
    """One guide page."""

    page_id: str
    title: str
    keywords: list[str]
    body: str

    def sections(self) -> list[Section]:
        """Split on `##` headings.

        Markdown headings are used as the boundary rather than a fixed token
        window because the author already decided where one topic ends -- that is
        what a heading is. A sliding window would cut 「端末を交換したとき」 in half
        and produce two chunks that each answer the question badly.

        Text before the first heading is kept as its own section. It is the page's
        preamble, it usually states who the page is for, and dropping it loses the
        only sentence that says the page exists.
        """
        keywords = "、".join(self.keywords)
        header = f"{self.title}\n{keywords}" if keywords else self.title

        found: list[Section] = []
        heading = "（前書き）"
        buffer: list[str] = []

        def flush() -> None:
            text = "\n".join(buffer).strip()
            if text:
                found.append(Section(self.page_id, header, heading, text))

        for line in self.body.splitlines():
            if line.startswith("## "):
                flush()
                heading = line[3:].strip()
                buffer = []
            else:
                buffer.append(line)
        flush()

        # A page with no headings still has to be searchable, so fall back to the
        # whole body rather than returning nothing and silently dropping the page
        # out of the index.
        if not found:
            found.append(Section(self.page_id, header, self.title, self.body))
        return found


@dataclass(frozen=True)
class Ticket:
    """One inbound question, with the label it will be scored against."""

    ticket_id: str
    text: str
    expect_cause: str
    needs_human: bool
    expect_page: str | None = None
    note: str = ""


def load_pages(directory: Path = GUIDE_DIR) -> list[Page]:
    """Read every markdown page, front matter first."""
    pages = []
    for path in sorted(directory.glob("*.md")):
        front, body = _split_front_matter(path.read_text(encoding="utf-8"))
        pages.append(
            Page(
                page_id=path.stem,
                title=front["title"],
                keywords=front.get("keywords", []),
                body=body.strip(),
            )
        )
    if not pages:
        raise FileNotFoundError(f"no guide pages under {directory}")
    return pages


def load_tickets(path: Path = TICKETS) -> list[Ticket]:
    """Read the labelled ticket batch."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        Ticket(
            ticket_id=item["id"],
            text=item["text"],
            expect_cause=item["expect_cause"],
            needs_human=item.get("needs_human", False),
            expect_page=item.get("expect_page"),
            note=item.get("note", ""),
        )
        for item in raw
    ]


def _split_front_matter(text: str) -> tuple[dict, str]:
    """Separate YAML front matter from the body.

    Raises rather than guessing when the delimiters are missing. A page whose
    front matter silently failed to parse would still embed -- as its own raw
    YAML -- and would then quietly rank badly, which is a bug that presents as a
    retrieval quality problem and wastes an afternoon.
    """
    if not text.startswith("---"):
        raise ValueError("page is missing YAML front matter")
    _, front, body = text.split("---", 2)
    parsed = yaml.safe_load(front)
    if "title" not in parsed:
        raise ValueError("page front matter must have a title")
    return parsed, body
