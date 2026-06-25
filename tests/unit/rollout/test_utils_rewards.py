import pytest

from osmosis_ai.rollout.types import RolloutSample
from osmosis_ai.rollout.utils.rewards import validate_sample_has_reward


class TestValidateSampleHasReward:
    def test_raises_when_no_sample(self):
        with pytest.raises(ValueError, match="No sample to grade"):
            validate_sample_has_reward(None)

    def test_raises_when_sample_has_no_reward(self):
        sample = RolloutSample(messages=[])

        with pytest.raises(ValueError, match="Sample has no reward after grading"):
            validate_sample_has_reward(sample)

    def test_accepts_sample_with_reward(self):
        validate_sample_has_reward(RolloutSample(messages=[], reward=1.0))

    def test_accepts_removed_sample_without_reward(self):
        # A sample flagged for removal is exempt from the reward requirement.
        validate_sample_has_reward(RolloutSample(messages=[], remove_sample=True))
