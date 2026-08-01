"""Generic per-key phase timing.

Mark named phases as they happen; finish() returns the durations between
consecutive marks. Phases appear in the order they were marked.
"""

from __future__ import annotations

import time


class PhaseTimer:
    def __init__(self) -> None:
        self.marks: dict[str, dict[str, float]] = {}

    def start(self, key: str) -> None:
        self.marks[key] = {"start": time.monotonic()}

    def mark(self, key: str, phase: str) -> None:
        marks = self.marks.get(key)
        if marks is not None:
            marks[phase] = time.monotonic()

    def finish(self, key: str) -> dict[str, float] | None:
        marks = self.marks.pop(key, None)
        if not marks:
            return None
        marks["end"] = time.monotonic()
        phases = list(marks)
        timings = {
            f"{a}->{b}": round(marks[b] - marks[a], 2)
            for a, b in zip(phases, phases[1:])
        }
        timings["total"] = round(marks["end"] - marks["start"], 2)
        return timings
