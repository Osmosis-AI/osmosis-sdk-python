"""RolloutSample: the reward must survive the trip to the controller.

A reward that serializes to JSON ``null`` — or that cannot be serialized at
all — reaches the controller as "no reward", which is indistinguishable from a
grader that never ran.
"""

import json
import math
import numbers

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


@numbers.Real.register
class fake_float32:
    """``np.float32`` stand-in: a Real that is not a builtin float.

    numpy registers its scalar types with the ``numbers`` ABCs exactly like
    this, which is what lets the sanitizer recognize them without importing
    numpy. ``json.dumps`` rejects such objects outright.
    """

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def __float__(self) -> float:
        return self.value


@numbers.Integral.register
class fake_int64:
    """``np.int64`` stand-in: an Integral that is not a builtin int."""

    def __init__(self, value: int) -> None:
        self.value = int(value)

    def __int__(self) -> int:
        return self.value

    def __index__(self) -> int:
        return self.value


class TestJsonSafeCopy:
    def test_metrics_mutated_in_place_are_still_sanitized(self):
        # No validator can intercept dict mutation, which is how graders write
        # metrics — so the wire boundary has to be the one that catches it.
        sample = RolloutSample(messages=[], reward=1.0)
        sample.metrics["good"] = 0.5
        sample.metrics["bad"] = float("nan")

        cleaned = sample.json_safe_copy()

        assert cleaned.metrics == {"good": 0.5}
        assert cleaned.reward == 1.0
        json.dumps(cleaned.model_dump())

    def test_nested_values_are_sanitized(self):
        sample = RolloutSample(messages=[])
        sample.extra_fields["nested"] = {"ok": 1.0, "bad": float("inf")}
        sample.extra_fields["列表"] = [1.0, float("-inf"), 2.0]

        cleaned = sample.json_safe_copy()

        assert cleaned.extra_fields == {"nested": {"ok": 1.0}, "列表": [1.0, 2.0]}
        json.dumps(cleaned.model_dump())

    def test_clean_sample_is_returned_unchanged(self):
        sample = RolloutSample(messages=[], reward=1.0)
        sample.metrics["fine"] = 0.25
        assert sample.json_safe_copy() is sample

    def test_non_float_values_are_preserved(self):
        sample = RolloutSample(messages=[])
        sample.metrics.update({"s": "text", "b": True, "n": None, "i": 3})
        assert sample.json_safe_copy().metrics == {
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

    def test_foreign_numeric_scalars_normalize_to_builtins(self):
        # np.float32/np.int64 pass an isinstance(value, float) gate untouched
        # and then json.dumps raises TypeError — losing the whole callback,
        # reward included. Normalize through the numbers ABCs instead.
        sample = RolloutSample(messages=[], reward=0.8)
        sample.metrics["f32"] = fake_float32(0.5)
        sample.metrics["i64"] = fake_int64(7)
        sample.metrics["nested"] = {"deep": [fake_float32(1.5), fake_int64(2)]}

        cleaned = sample.json_safe_copy()

        assert cleaned.metrics == {
            "f32": 0.5,
            "i64": 7,
            "nested": {"deep": [1.5, 2]},
        }
        assert type(cleaned.metrics["f32"]) is float
        assert type(cleaned.metrics["i64"]) is int
        assert cleaned.reward == 0.8
        json.dumps(cleaned.model_dump(), allow_nan=False)

    def test_non_finite_foreign_scalars_are_dropped(self):
        sample = RolloutSample(messages=[])
        sample.metrics["bad"] = fake_float32(float("inf"))
        sample.metrics["good"] = fake_float32(1.0)

        assert sample.json_safe_copy().metrics == {"good": 1.0}

    def test_unencodable_objects_and_keys_are_dropped(self):
        sample = RolloutSample(messages=[], reward=0.8)
        sample.metrics["obj"] = object()
        sample.metrics["bytes"] = b"raw"
        sample.metrics["ok"] = 1
        sample.extra_fields[("tuple", "key")] = "dropped with its key"  # type: ignore[index]
        sample.extra_fields[fake_int64(3)] = "numeric key normalizes"  # type: ignore[index]

        cleaned = sample.json_safe_copy()

        assert cleaned.metrics == {"ok": 1}
        assert cleaned.extra_fields == {"3": "numeric key normalizes"}
        assert cleaned.reward == 0.8
        json.dumps(cleaned.model_dump(), allow_nan=False)

    def test_real_numpy_scalars_if_installed(self):
        np = pytest.importorskip("numpy")
        sample = RolloutSample(messages=[], reward=0.8)
        sample.metrics.update(
            {
                "f16": np.float16(0.5),
                "f32": np.float32(0.5),
                "f64": np.float64(0.5),
                "i32": np.int32(7),
                "i64": np.int64(7),
                "nan32": np.float32("nan"),
            }
        )

        cleaned = sample.json_safe_copy()

        assert cleaned.metrics == {
            "f16": 0.5,
            "f32": 0.5,
            "f64": 0.5,
            "i32": 7,
            "i64": 7,
        }
        assert all(type(v) in (int, float) for v in cleaned.metrics.values()), (
            "every numpy scalar must become a builtin"
        )
        assert cleaned.reward == 0.8
        json.dumps(cleaned.model_dump(), allow_nan=False)
