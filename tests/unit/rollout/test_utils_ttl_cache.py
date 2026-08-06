"""TtlCache: uniform-TTL dict with tombstone-based amortized expiry."""

from unittest.mock import patch

from osmosis_ai.rollout.utils.ttl_cache import TtlCache


def test_set_get_and_expiry():
    clock = [0.0]
    with patch("osmosis_ai.rollout.utils.ttl_cache.time.monotonic", lambda: clock[0]):
        cache = TtlCache(ttl_sec=10.0)
        cache.set("a", 1)
        assert cache.get("a") == 1
        clock[0] = 9.9
        assert cache.get("a") == 1
        clock[0] = 10.0
        assert cache.get("a") is None


def test_writes_prune_expired_entries():
    clock = [0.0]
    with patch("osmosis_ai.rollout.utils.ttl_cache.time.monotonic", lambda: clock[0]):
        cache = TtlCache(ttl_sec=10.0)
        cache.set("a", 1)
        cache.set("b", 2)
        clock[0] = 11.0
        cache.set("c", 3)
        assert len(cache) == 1
        assert not cache.tombstones or cache.tombstones[0][1] == "c"


def test_overwrite_refreshes_and_orphans_old_tombstone():
    clock = [0.0]
    with patch("osmosis_ai.rollout.utils.ttl_cache.time.monotonic", lambda: clock[0]):
        cache = TtlCache(ttl_sec=10.0)
        cache.set("a", 1)
        clock[0] = 5.0
        cache.set("a", 2)
        # Past the first deadline: the stale tombstone must not evict the refresh.
        clock[0] = 12.0
        cache.set("other", 3)
        assert cache.get("a") == 2
        clock[0] = 15.0
        assert cache.get("a") is None
