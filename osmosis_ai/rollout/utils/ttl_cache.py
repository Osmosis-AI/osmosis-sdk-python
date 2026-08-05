"""Dict with a uniform per-entry TTL and amortized O(1) expiry.

Uniform TTLs expire in insertion order, so a deque of (deadline, key)
tombstones is pruned from the head on every write. The deadline doubles as
the tombstone marker: overwriting a key refreshes its deadline, which
orphans the old tombstone harmlessly.
"""

from __future__ import annotations

import time
from collections import deque


class TtlCache[K, V]:
    def __init__(self, ttl_sec: float) -> None:
        self.ttl_sec = ttl_sec
        self.entries: dict[K, tuple[float, V]] = {}
        self.tombstones: deque[tuple[float, K]] = deque()

    def set(self, key: K, value: V) -> None:
        now = time.monotonic()
        while self.tombstones and self.tombstones[0][0] <= now:
            deadline, stale = self.tombstones.popleft()
            entry = self.entries.get(stale)
            if entry is not None and entry[0] == deadline:
                del self.entries[stale]
        deadline = now + self.ttl_sec
        self.entries[key] = (deadline, value)
        self.tombstones.append((deadline, key))

    def get(self, key: K) -> V | None:
        entry = self.entries.get(key)
        if entry is None or entry[0] <= time.monotonic():
            return None
        return entry[1]

    def __len__(self) -> int:
        return len(self.entries)
