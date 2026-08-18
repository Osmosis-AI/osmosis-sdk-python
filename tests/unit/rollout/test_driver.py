from osmosis_ai.rollout import driver
from osmosis_ai.rollout.driver import RolloutDriver, RolloutOutcome, RolloutRunRequest
from osmosis_ai.rollout.types import RolloutStatus


def test_rollout_outcome_defaults():
    outcome = RolloutOutcome(status=RolloutStatus.SUCCESS)
    assert outcome.status == RolloutStatus.SUCCESS
    assert outcome.sample is None
    assert outcome.error is None
    assert outcome.rollout_id is None


def test_driver_exports_controller_contract_only():
    assert driver.__all__ == ["RolloutDriver", "RolloutOutcome", "RolloutRunRequest"]
    assert not hasattr(driver, "InProcessDriver")


def test_rollout_run_request_defaults():
    request = RolloutRunRequest(messages=[{"role": "user", "content": "hi"}])
    assert request.label is None
    assert request.metadata is None
    assert request.rollout_id == ""
    assert request.agent_timeout_sec is None
    assert request.grader_timeout_sec is None
    assert request.extra_fields is None


async def test_driver_run_accepts_a_single_request():
    class RecordingDriver(RolloutDriver):
        def __init__(self) -> None:
            self.seen: RolloutRunRequest | None = None

        async def run(self, request: RolloutRunRequest) -> RolloutOutcome:
            self.seen = request
            return RolloutOutcome(
                status=RolloutStatus.SUCCESS, rollout_id=request.rollout_id
            )

    request = RolloutRunRequest(
        messages=[{"role": "user", "content": "hi"}],
        label="yes",
        metadata={"row_index": 3},
        rollout_id="abc123",
        agent_timeout_sec=30.0,
        grader_timeout_sec=10.0,
        extra_fields={"run_index": 0},
    )
    recorded = RecordingDriver()
    outcome = await recorded.run(request)
    assert recorded.seen is request
    assert outcome.rollout_id == "abc123"
