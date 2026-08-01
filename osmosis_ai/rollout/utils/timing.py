"""Generic per-key phase timing.

mark() records that a named phase begins; finish() reports each phase's
duration (from its mark to the next mark, or to finish() for the last one).
"""

from __future__ import annotations

import time


class PhaseTimer:
    def __init__(self) -> None:
        self.marks: dict[str, list[tuple[str, float]]] = {}

    def start(self, key: str) -> None:
        self.marks[key] = [("start", time.monotonic())]

    def mark(self, key: str, phase: str) -> None:
        marks = self.marks.get(key)
        if marks is not None:
            marks.append((phase, time.monotonic()))

    def finish(self, key: str) -> dict[str, float] | None:
        marks = self.marks.pop(key, None)
        if not marks:
            return None
        end = time.monotonic()
        timings = {
            phase: round(next_at - at, 2)
            for (phase, at), (_, next_at) in zip(marks, [*marks[1:], ("end", end)])
            if phase != "start"
        }
        timings["total"] = round(end - marks[0][1], 2)
        return timings
