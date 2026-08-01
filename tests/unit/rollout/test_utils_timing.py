"""PhaseTimer: phases are named for the work running during each span."""

from unittest.mock import patch

from osmosis_ai.rollout.utils.timing import PhaseTimer


def test_phases_named_after_running_phase():
    clock = iter([0.0, 1.0, 94.0, 97.0, 101.0])
    with patch("osmosis_ai.rollout.utils.timing.time.monotonic", lambda: next(clock)):
        timer = PhaseTimer()
        timer.start("r1")
        timer.mark("r1", "environment")
        timer.mark("r1", "agent")
        timer.mark("r1", "verification")
        timings = timer.finish("r1")

    assert timings == {
        "environment": 93.0,
        "agent": 3.0,
        "verification": 4.0,
        "total": 101.0,
    }


def test_unknown_key_is_ignored():
    timer = PhaseTimer()
    timer.mark("missing", "environment")
    assert timer.finish("missing") is None
