from osmosis_ai.rollout.types import RolloutSample


def validate_samples_have_rewards(samples: dict[str, RolloutSample]) -> None:
    if not samples:
        raise ValueError("No samples after grading")

    for sid, sample in samples.items():
        if sample.reward is None:
            raise ValueError(f"Sample {sid} has no reward after grading")
