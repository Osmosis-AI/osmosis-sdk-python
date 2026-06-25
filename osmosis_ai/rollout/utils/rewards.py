from osmosis_ai.rollout.types import RolloutSample


def validate_sample_has_reward(sample: RolloutSample | None) -> None:
    """Raise if the rollout did not produce a graded sample.

    A rollout has at most one sample; an ungraded or missing sample is a
    grader contract violation (or a workflow that forgot to register an
    agent/session, in which case ``sample`` is ``None``).
    """
    if sample is None:
        raise ValueError(
            "No sample to grade. The workflow must register a sample "
            "source (e.g. via OsmosisStrandsAgent or OsmosisMemorySession) "
            "before grading."
        )
    if not sample.remove_sample and sample.reward is None:
        raise ValueError("Sample has no reward after grading")
