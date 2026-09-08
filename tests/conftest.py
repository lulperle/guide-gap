"""Shared fakes. No credentials, no network, in any test in this suite.

That is a design constraint rather than a testing preference: every decision worth
testing here -- ranking, thresholds, cluster membership, reply parsing -- is a pure
function of vectors and strings, and the AWS call is one thin file. A suite that
needed Bedrock would run nowhere except a laptop with a session token, which is the
same as not running.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeClient:
    """Stands in for `Bedrock`, returning canned replies in order.

    Duck-typed rather than a subclass, because `Bedrock.__init__` builds a boto3
    client and a test that has to avoid its own constructor is testing the mock.
    """

    def __init__(self, replies: list[str] | None = None) -> None:
        self.replies = list(replies or [])
        self.prompts: list[tuple[str, str, int]] = []
        self.embedded: list[str] = []

    def draft(self, system: str, prompt: str, max_tokens: int = 2000) -> str:
        self.prompts.append((system, prompt, max_tokens))
        return self.replies.pop(0) if self.replies else ""

    def embed(self, text: str) -> list[float]:
        """A deterministic pseudo-embedding.

        Character-bucket counts, not a real model: the point is that identical text
        gives an identical vector and overlapping text gives a nearer one, which is
        all the code above this layer relies on.
        """
        buckets = [0.0] * 16
        for character in text:
            buckets[ord(character) % 16] += 1.0
        self.embedded.append(text)
        return buckets

    @property
    def calls(self) -> int:
        return len(self.prompts)


@pytest.fixture
def fake() -> FakeClient:
    return FakeClient()


def vector(*values: float) -> list[float]:
    """A short vector. Dimensionality is irrelevant to every function under test."""
    return list(values)
