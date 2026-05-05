import pytest

from osmosis_ai.rollout.types import RolloutSample
from osmosis_ai.rollout.utils.rewards import validate_samples_have_rewards


class TestValidateSamplesHaveRewards:
    def test_raises_when_no_samples(self):
        with pytest.raises(ValueError, match="No samples after grading"):
            validate_samples_have_rewards({})

    def test_raises_when_sample_has_no_reward(self):
        samples = {
            "sample-1": RolloutSample(id="sample-1", messages=[], reward=1.0),
            "sample-2": RolloutSample(id="sample-2", messages=[]),
        }

        with pytest.raises(
            ValueError, match="Sample sample-2 has no reward after grading"
        ):
            validate_samples_have_rewards(samples)

    def test_accepts_samples_with_rewards(self):
        samples = {
            "sample-1": RolloutSample(id="sample-1", messages=[], reward=1.0),
            "sample-2": RolloutSample(id="sample-2", messages=[], reward=0.0),
        }

        validate_samples_have_rewards(samples)
