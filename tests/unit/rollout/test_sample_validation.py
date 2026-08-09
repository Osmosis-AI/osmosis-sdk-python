"""RolloutSample: the reward must survive the trip to the controller.

A reward that serializes to JSON ``null`` — or that cannot be serialized at
all — reaches the controller as "no reward", which is indistinguishable from a
grader that never ran.
"""

import json
import math

import pytest
from pydantic import ValidationError

from osmosis_ai.rollout.context import GraderContext
from osmosis_ai.rollout.types import RolloutSample

NON_FINITE = [float("nan"), float("inf"), float("-inf")]


class TestRewardDomain:
    @pytest.mark.parametrize("value", NON_FINITE)
    def test_construction_rejects_non_finite(self, value):
        with pytest.raises(ValidationError):
            RolloutSample(messages=[], reward=value)

    @pytest.mark.parametrize("value", NON_FINITE)
    def test_assignment_rejects_non_finite(self, value):
        sample = RolloutSample(messages=[])
        with pytest.raises(ValidationError):
            sample.reward = value

    def test_assignment_rejects_non_numeric(self):
        sample = RolloutSample(messages=[])
        with pytest.raises(ValidationError):
            sample.reward = "not-a-number"  # type: ignore[assignment]

    @pytest.mark.parametrize("value,expected", [(1, 1.0), (0.5, 0.5), (-2, -2.0)])
    def test_finite_numbers_are_accepted(self, value, expected):
        sample = RolloutSample(messages=[])
        sample.reward = value
        assert sample.reward == expected

    def test_none_stays_allowed(self):
        # "Not graded yet" is a legitimate state; only non-finite is not.
        assert RolloutSample(messages=[]).reward is None

    @pytest.mark.parametrize("value", NON_FINITE)
    def test_set_reward_rejects_non_finite(self, value):
        # The path every grader actually takes.
        ctx = GraderContext(label="x", sample=RolloutSample(messages=[]))
        with pytest.raises(ValidationError):
            ctx.set_reward(value)

    def test_trajectory_snapshot_still_defaults(self):
        # validate_assignment re-enters the after-validator; the default must
        # still be computed exactly once and an explicit None still disable it.
        messages = [{"role": "user", "content": "hi"}]
        assert RolloutSample(messages=messages).trajectory_messages == messages
        assert (
            RolloutSample(
                messages=messages, trajectory_messages=None
            ).trajectory_messages
            is None
        )


class TestDropNonFiniteValues:
    def test_metrics_mutated_in_place_are_still_sanitized(self):
        # No validator can intercept dict mutation, which is how graders write
        # metrics — so the wire boundary has to be the one that catches it.
        sample = RolloutSample(messages=[], reward=1.0)
        sample.metrics["good"] = 0.5
        sample.metrics["bad"] = float("nan")

        cleaned = sample.drop_non_finite_values()

        assert cleaned.metrics == {"good": 0.5}
        assert cleaned.reward == 1.0
        json.dumps(cleaned.model_dump())

    def test_nested_values_are_sanitized(self):
        sample = RolloutSample(messages=[])
        sample.extra_fields["nested"] = {"ok": 1.0, "bad": float("inf")}
        sample.extra_fields["列表"] = [1.0, float("-inf"), 2.0]

        cleaned = sample.drop_non_finite_values()

        assert cleaned.extra_fields == {"nested": {"ok": 1.0}, "列表": [1.0, 2.0]}
        json.dumps(cleaned.model_dump())

    def test_clean_sample_is_returned_unchanged(self):
        sample = RolloutSample(messages=[], reward=1.0)
        sample.metrics["fine"] = 0.25
        assert sample.drop_non_finite_values() is sample

    def test_non_float_values_are_preserved(self):
        sample = RolloutSample(messages=[])
        sample.metrics.update({"s": "text", "b": True, "n": None, "i": 3})
        assert sample.drop_non_finite_values().metrics == {
            "s": "text",
            "b": True,
            "n": None,
            "i": 3,
        }

    def test_json_dumps_would_otherwise_emit_invalid_json(self):
        # Why this exists at all: the stdlib happily writes a literal no strict
        # parser accepts, and HTTPX's encoder raises outright.
        assert json.dumps({"x": float("nan")}) == '{"x": NaN}'
        with pytest.raises(ValueError):
            json.dumps({"x": float("nan")}, allow_nan=False)
        assert not math.isfinite(float("nan"))
