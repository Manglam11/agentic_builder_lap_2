"""Long-term memory — persist facts, recall by meaning.

embed -> store -> cosine recall, ported from the Module 3 puzzles. This is the
retrieval artifact: be ready to defend save() and recall() line by line.
"""

import math

from agentic_track.llm import embed


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class MemoryStore:
    """In-memory for now; swap _items for a vector DB to survive restarts."""

    def __init__(self):
        self._items = []  # each: {"text": str, "vec": list[float]}

    def save(self, text):
        self._items.append({"text": text, "vec": embed(text)})

    def recall(self, query, k=3):
        if not self._items:
            return []  # empty store fails soft, not with a crash
        qv = embed(query)
        ranked = sorted(
            self._items,
            key=lambda m: cosine(qv, m["vec"]),
            reverse=True,  # most similar first — the Puzzle 5 trap
        )
        return [m["text"] for m in ranked[:k]]
