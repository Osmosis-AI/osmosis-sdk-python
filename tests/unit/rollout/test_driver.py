from osmosis_ai.rollout import driver
from osmosis_ai.rollout.driver import RolloutOutcome
from osmosis_ai.rollout.types import RolloutStatus


def test_rollout_outcome_defaults():
    outcome = RolloutOutcome(status=RolloutStatus.SUCCESS)
    assert outcome.status == RolloutStatus.SUCCESS
    assert outcome.samples == {}
    assert outcome.error is None
    assert outcome.duration_ms == 0.0
    assert outcome.tokens == 0
    assert outcome.systemic_error is None


def test_driver_exports_controller_contract_only():
    assert driver.__all__ == ["RolloutDriver", "RolloutOutcome"]
    assert not hasattr(driver, "InProcessDriver")
