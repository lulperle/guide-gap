"""On-disk cache for embeddings.

Embeddings of a fixed corpus are deterministic, so paying for them twice is
waste -- but the real reason this exists is that the thresholds in `coverage.py`
and `cluster.py` are judgement calls that need to be re-examined against the
whole labelled set. Without a cache, every look at a threshold costs a full
embedding pass, and the honest response to that cost is to stop looking, which
is how a magic number survives.

Keyed by the model id as well as the text. Two models' vectors are not
interchangeable, and a cache that silently mixes them produces similarity scores
that are arithmetic noise -- the worst class of bug here, because nothing errors.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class EmbeddingCache:
    """A text -> vector store backed by one JSON file."""

    def __init__(self, path: Path, *, model: str):
        self.path = path
        self.model = model
        self._data: dict[str, list[float]] = {}
        self._dirty = False
        if path.exists():
            self._data = json.loads(path.read_text(encoding="utf-8"))

    def key(self, text: str) -> str:
        digest = hashlib.sha256(f"{self.model}\n{text}".encode()).hexdigest()
        return digest[:32]

    def get(self, text: str) -> list[float] | None:
        return self._data.get(self.key(text))

    def put(self, text: str, vector: list[float]) -> None:
        self._data[self.key(text)] = vector
        self._dirty = True

    def save(self) -> None:
        """Write only if something changed, so a read-only run touches nothing."""
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data), encoding="utf-8")
        self._dirty = False

    def __len__(self) -> int:
        return len(self._data)
