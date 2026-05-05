import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock

from osmosis_ai.rollout.backend.harbor.backend import HarborBackend, PendingTrial
from osmosis_ai.rollout.types import (
    RolloutErrorCategory,
    RolloutSample,
    RolloutStatus,
)


class TestHarborBackend:
    async def test_empty_verifier_rewards_logs_and_returns_validation_failure(
        self, caplog
    ):
        backend = HarborBackend.__new__(HarborBackend)
        backend.pending = {}
        backend.cleanup_successful_trials = False

        on_workflow = AsyncMock()
        on_grader = AsyncMock()
        pending = PendingTrial(on_workflow, on_grader)
        pending.workflow_complete_called = True
        backend.pending["r1"] = pending

        event = SimpleNamespace(
            config=SimpleNamespace(trial_name="trial-r1"),
            result=SimpleNamespace(
                agent_result=SimpleNamespace(
                    metadata={
                        "status": "success",
                        "samples": {
                            "sample-1": RolloutSample(
                                id="sample-1", messages=[]
                            ).model_dump()
                        },
                    }
                ),
                verifier_result=SimpleNamespace(rewards={}),
                exception_info=None,
            ),
        )

        with caplog.at_level(
            logging.WARNING, logger="osmosis_ai.rollout.backend.harbor.backend"
        ):
            await backend.on_trial_end(event)

        on_grader.assert_awaited_once()
        result = on_grader.call_args.args[0]
        assert result.status == RolloutStatus.FAILURE
        assert result.err_category == RolloutErrorCategory.VALIDATION_ERROR
        assert "Harbor verifier returned empty rewards for rollout r1" in caplog.text
