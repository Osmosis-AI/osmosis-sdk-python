import pytest

from osmosis_ai.rollout.driver import InProcessDriver, RolloutOutcome
from osmosis_ai.rollout.types import (
    ExecutionResult,
    RolloutSample,
    RolloutStatus,
)


def test_rollout_outcome_defaults():
    outcome = RolloutOutcome(status=RolloutStatus.SUCCESS)
    assert outcome.status == RolloutStatus.SUCCESS
    assert outcome.samples == {}
    assert outcome.error is None
    assert outcome.duration_ms == 0.0
    assert outcome.tokens == 0
    assert outcome.systemic_error is None


class FakeProxy:
    """Minimal proxy stub for InProcessDriver tests."""

    def __init__(self, systemic_errors: dict[str, str] | None = None):
        self.url = "http://127.0.0.1:9999/v1"
        self.api_key = "test-key"
        self._tokens: dict[str, int] = {}
        self._systemic_errors: dict[str, str] = systemic_errors or {}

    def collect_tokens(self, rollout_id: str) -> int:
        return self._tokens.pop(rollout_id, 0)

    def collect_systemic_error(self, rollout_id: str) -> str | None:
        return self._systemic_errors.pop(rollout_id, None)

    def set_tokens(self, rollout_id: str, count: int):
        self._tokens[rollout_id] = count


class FakeBackend:
    """Minimal ExecutionBackend stub."""

    def __init__(self, workflow_result=None, grader_result=None, raise_error=None):
        self._workflow_result = workflow_result
        self._grader_result = grader_result
        self._raise_error = raise_error
        self.max_concurrency = 0
        self.last_request = None

    async def execute(self, request, on_workflow_complete, on_grader_complete=None):
        self.last_request = request
        if self._raise_error:
            raise self._raise_error
        if self._workflow_result:
            await on_workflow_complete(self._workflow_result)
        if self._grader_result and on_grader_complete:
            await on_grader_complete(self._grader_result)


@pytest.fixture
def success_backend():
    return FakeBackend(
        workflow_result=ExecutionResult(
            status=RolloutStatus.SUCCESS,
            samples={
                "s1": RolloutSample(
                    id="s1",
                    messages=[{"role": "assistant", "content": "hello"}],
                    reward=0.8,
                )
            },
        ),
    )


@pytest.fixture
def graded_backend():
    return FakeBackend(
        workflow_result=ExecutionResult(
            status=RolloutStatus.SUCCESS,
            samples={
                "s1": RolloutSample(
                    id="s1",
                    messages=[{"role": "assistant", "content": "hello"}],
                )
            },
        ),
        grader_result=ExecutionResult(
            status=RolloutStatus.SUCCESS,
            samples={
                "s1": RolloutSample(
                    id="s1",
                    messages=[{"role": "assistant", "content": "hello"}],
                    reward=0.95,
                )
            },
        ),
    )


async def test_in_process_driver_success(success_backend):
    proxy = FakeProxy()
    proxy.set_tokens("eval-0-run-0", 150)
    driver = InProcessDriver(backend=success_backend, proxy=proxy)

    outcome = await driver.run(
        messages=[{"role": "user", "content": "test"}],
        label="expected",
        rollout_id="eval-0-run-0",
    )

    assert outcome.status == RolloutStatus.SUCCESS
    assert "s1" in outcome.samples
    assert outcome.samples["s1"].reward == 0.8
    assert outcome.tokens == 150
    assert outcome.duration_ms > 0
    assert outcome.systemic_error is None
    assert outcome.rollout_id == "eval-0-run-0"
    assert outcome.full_callback_sample_ids == ["s1"]
    assert outcome.scored_sample_ids == ["s1"]
    assert outcome.skipped_sample_ids == []
    assert outcome.skipped is False


async def test_in_process_driver_graded(graded_backend):
    proxy = FakeProxy()
    driver = InProcessDriver(backend=graded_backend, proxy=proxy)

    outcome = await driver.run(
        messages=[{"role": "user", "content": "test"}],
        label="expected",
        rollout_id="eval-0-run-0",
    )

    assert outcome.status == RolloutStatus.SUCCESS
    assert outcome.samples["s1"].reward == 0.95  # grader result takes precedence


async def test_in_process_driver_generates_rollout_id_when_blank(success_backend):
    proxy = FakeProxy()
    driver = InProcessDriver(backend=success_backend, proxy=proxy)

    outcome = await driver.run(
        messages=[{"role": "user", "content": "test"}],
        label="expected",
        rollout_id="",
    )

    assert outcome.rollout_id
    assert outcome.rollout_id.startswith("in-process-")
    assert success_backend.last_request.id == outcome.rollout_id


async def test_in_process_driver_failed_removed_samples_are_not_skipped():
    backend = FakeBackend(
        workflow_result=ExecutionResult(
            status=RolloutStatus.FAILURE,
            samples={
                "s1": RolloutSample(
                    id="s1",
                    remove_sample=True,
                )
            },
            err_message="grader failed",
        ),
    )
    proxy = FakeProxy()
    driver = InProcessDriver(backend=backend, proxy=proxy)

    outcome = await driver.run(
        messages=[{"role": "user", "content": "test"}],
        rollout_id="eval-0-run-0",
    )

    assert outcome.status == RolloutStatus.FAILURE
    assert outcome.error == "grader failed"
    assert outcome.scored_sample_ids == []
    assert outcome.skipped_sample_ids == ["s1"]
    assert outcome.skipped is False


async def test_in_process_driver_backend_exception():
    backend = FakeBackend(raise_error=RuntimeError("workflow crashed"))
    proxy = FakeProxy()
    driver = InProcessDriver(backend=backend, proxy=proxy)

    outcome = await driver.run(
        messages=[{"role": "user", "content": "test"}],
        rollout_id="eval-0-run-0",
    )

    assert outcome.status == RolloutStatus.FAILURE
    assert "workflow crashed" in outcome.error
    assert outcome.duration_ms > 0


async def test_in_process_driver_systemic_error():
    backend = FakeBackend(
        workflow_result=ExecutionResult(
            status=RolloutStatus.FAILURE, err_message="auth fail"
        ),
    )
    proxy = FakeProxy(systemic_errors={"eval-0-run-0": "Authentication failed"})
    driver = InProcessDriver(backend=backend, proxy=proxy)

    outcome = await driver.run(
        messages=[{"role": "user", "content": "test"}],
        rollout_id="eval-0-run-0",
    )

    assert outcome.systemic_error == "Authentication failed"
    assert outcome.error == "auth fail"


async def test_in_process_driver_max_concurrency():
    backend = FakeBackend()
    backend.max_concurrency = 8
    proxy = FakeProxy()
    driver = InProcessDriver(backend=backend, proxy=proxy)
    assert driver.max_concurrency == 8


async def test_in_process_driver_no_result():
    """Backend completes without calling any callback."""
    backend = FakeBackend()  # No results configured
    proxy = FakeProxy()
    driver = InProcessDriver(backend=backend, proxy=proxy)

    outcome = await driver.run(
        messages=[{"role": "user", "content": "test"}],
        rollout_id="eval-0-run-0",
    )

    assert outcome.status == RolloutStatus.FAILURE
    assert "No result from backend" in outcome.error
