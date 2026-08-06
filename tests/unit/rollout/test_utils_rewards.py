import pytest

from osmosis_ai.rollout.types import RolloutSample
from osmosis_ai.rollout.utils.rewards import validate_sample_has_reward


class TestValidateSampleHasReward:
    def test_raises_when_no_sample(self):
        with pytest.raises(ValueError, match="No sample to grade"):
            validate_sample_has_reward(None)

    def test_raises_when_sample_has_no_reward(self):
        with pytest.raises(ValueError, match="Sample has no reward after grading"):
            validate_sample_has_reward(RolloutSample(messages=[]))

    def test_accepts_sample_with_reward(self):
        validate_sample_has_reward(RolloutSample(messages=[], reward=1.0))

    def test_accepts_removed_sample_without_reward(self):
        validate_sample_has_reward(RolloutSample(messages=[], remove_sample=True))


class TestPickReward:
    def test_named_key_wins(self):
        from osmosis_ai.rollout.utils.rewards import pick_reward

        assert pick_reward({"reward": 1.0, "style": 0.2}, "reward") == 1.0
        assert pick_reward({"accuracy": 0.5}, "accuracy") == 0.5

    def test_sole_channel_is_unambiguous(self):
        from osmosis_ai.rollout.utils.rewards import pick_reward

        assert pick_reward({"accuracy": 0.75}, "reward") == 0.75

    def test_ambiguous_channels_yield_none(self, caplog):
        from osmosis_ai.rollout.utils.rewards import pick_reward

        assert pick_reward({"a": 1.0, "b": 0.0}, "reward") is None
        assert any("carry no 'reward'" in r.getMessage() for r in caplog.records)

    def test_empty_rewards_yield_none(self):
        from osmosis_ai.rollout.utils.rewards import pick_reward

        assert pick_reward({}, "reward") is None
        assert pick_reward(None, "reward") is None
