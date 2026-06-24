"""Unit tests for the NativeHarborBackend.

The harbor ``Trial`` is monkeypatched so these run without Docker/harbor task
resolution. The load-bearing assertion is the sample_id round-trip: the value
injected as the ``x-sample-id`` header must equal the key of the returned
``ExecutionResult.samples`` dict (which is the reward key the policy controller
reconciles against).
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest

from osmosis_ai.rollout.backend.base import ExecutionBackend
from osmosis_ai.rollout.backend.native_harbor import backend as bmod
from osmosis_ai.rollout.backend.native_harbor.backend import (
    NativeHarborBackend,
    resolve_task,
)
from osmosis_ai.rollout.context import RolloutContext
from osmosis_ai.rollout.types import (
    ExecutionRequest,
    RolloutErrorCategory,
    RolloutStatus,
)


def _trial_result(
    rewards: dict[str, float | int] | None = None,
    *,
    steps: list[dict[str, float | int]] | None = None,
    exc_message: str | None = None,
    exc_type: str | None = None,
) -> Any:
    """A duck-typed harbor TrialResult."""
    top = SimpleNamespace(rewards=rewards) if rewards is not None else None
    step_results = (
        [SimpleNamespace(verifier_result=SimpleNamespace(rewards=s)) for s in steps]
        if steps is not None
        else None
    )
    exception_info = (
        SimpleNamespace(exception_message=exc_message, exception_type=exc_type)
        if exc_message is not None
        else None
    )
    return SimpleNamespace(
        verifier_result=top,
        step_results=step_results,
        exception_info=exception_info,
    )


def _patch_trial(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: Any = None,
    create_error: Exception | None = None,
    capture: dict[str, Any] | None = None,
) -> None:
    """Replace the queue's submit() with a fake that records the TrialConfig it
    sees and returns a duck-typed TrialResult (so no harbor/Docker is needed)."""

    async def _submit(self: Any, trial_config: Any) -> SimpleNamespace:
        if capture is not None:
            capture["config"] = trial_config
        if create_error is not None:
            raise create_error
        return result if result is not None else _trial_result(rewards={"reward": 1.0})

    monkeypatch.setattr(bmod.TrialQueue, "submit", _submit)


def _request(metadata: dict[str, Any] | None = None, **kw: Any) -> ExecutionRequest:
    md = {"harbor_task": "/tmp/task"} if metadata is None else metadata
    return ExecutionRequest(
        id="ROLL", prompt=[{"role": "user", "content": "hi"}], metadata=md, **kw
    )


def _ctx() -> RolloutContext:
    return RolloutContext(
        chat_completions_url="http://ctrl:8080", api_key="sk-test", rollout_id="ROLL"
    )


class TestResolveTask:
    def test_local_path(self):
        cfg = resolve_task(_request({"harbor_task": "/tmp/some/task"}))
        assert cfg.path == Path("/tmp/some/task")
        assert cfg.name is None

    def test_package_with_ref(self):
        cfg = resolve_task(_request({"harbor_task": "harbor/hello-world@3"}))
        assert cfg.name == "harbor/hello-world"
        assert cfg.ref == "3"
        assert cfg.path is None

    def test_package_defaults_ref_latest(self):
        cfg = resolve_task(_request({"harbor_task": "harbor/hello-world"}))
        assert cfg.ref == "latest"

    def test_git_form(self):
        cfg = resolve_task(
            _request(
                {
                    "harbor_task": "git",
                    "git_url": "https://example.com/r.git",
                    "task_path": "tasks/foo",
                    "git_commit_id": "abc123",
                }
            )
        )
        assert cfg.git_url == "https://example.com/r.git"
        assert cfg.path == Path("tasks/foo")
        assert cfg.git_commit_id == "abc123"

    def test_missing_raises(self):
        with pytest.raises(ValueError, match="harbor_task"):
            resolve_task(_request({}))

    def test_bare_package_name_without_org_raises(self):
        # Not a path, not git, and missing the org/ slash -> clear error instead
        # of a cryptic downstream "not enough values to unpack".
        with pytest.raises(ValueError, match="org/name"):
            resolve_task(_request({"harbor_task": "helloworld"}))


class TestAgentConfig:
    def test_in_process_terminus2_nesting(self):
        backend = NativeHarborBackend()
        ac = backend._build_agent_config(_request(), _ctx(), "SID")
        assert ac.name == "terminus-2"
        # api_base is top-level; api_key/extra_headers ride inside llm_kwargs.
        assert ac.kwargs["api_base"] == "http://ctrl:8080"
        assert ac.kwargs["collect_rollout_details"] is False
        assert ac.kwargs["llm_kwargs"]["api_key"] == "sk-test"
        assert ac.kwargs["llm_kwargs"]["extra_headers"] == {
            "x-rollout-id": "ROLL",
            "x-sample-id": "SID",
        }
        # training-safety knobs forced for Terminus2.
        assert ac.kwargs["enable_summarize"] is False
        assert ac.kwargs["proactive_summarization_threshold"] == 0

    def test_inject_identity_headers_off(self):
        backend = NativeHarborBackend(inject_identity_headers=False)
        ac = backend._build_agent_config(_request(), _ctx(), "SID")
        assert "extra_headers" not in ac.kwargs.get("llm_kwargs", {})

    def test_training_safe_off_omits_summarize_knobs(self):
        backend = NativeHarborBackend(training_safe=False)
        ac = backend._build_agent_config(_request(), _ctx(), "SID")
        assert "enable_summarize" not in ac.kwargs

    def test_summarize_knobs_follow_resolved_class_not_name(self):
        # A customer wiring Terminus2 (or a subclass) via import_path has
        # name=None, yet the knobs must still be forced: the capability is read
        # from the resolved class, not the literal "terminus-2" string. The old
        # name-gate silently missed this -> training-unsafe rollouts.
        backend = NativeHarborBackend()
        md = {
            "harbor_task": "/tmp/task",
            "harbor_agent_import_path": "harbor.agents.terminus_2:Terminus2",
        }
        ac = backend._build_agent_config(_request(md), _ctx(), "SID")
        assert ac.name is None
        assert ac.kwargs["enable_summarize"] is False
        assert ac.kwargs["proactive_summarization_threshold"] == 0

    def test_in_process_agent_without_summarize_knobs(self):
        # An in-process custom agent whose constructor does not declare the knobs
        # is left alone -- forcing undeclared kwargs would raise inside harbor's
        # factory, so a drop-in agent without summarization stays runnable.
        backend = NativeHarborBackend()
        md = {
            "harbor_task": "/tmp/task",
            "harbor_agent_import_path": "harbor.agents.nop:NopAgent",
        }
        ac = backend._build_agent_config(_request(md), _ctx(), "SID")
        assert "enable_summarize" not in ac.kwargs

    def test_in_process_agent_missing_wiring_contract_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # A custom in-process agent whose constructor neither declares the
        # endpoint/identity kwargs nor accepts **kwargs is rejected at build time
        # with an actionable error, not a cryptic TypeError from harbor's factory.
        class _StrictAgent:
            def __init__(self, logs_dir, model_name=None):  # no wiring params
                pass

        monkeypatch.setattr(bmod, "_resolve_agent_class", lambda name, ip: _StrictAgent)
        backend = NativeHarborBackend()
        md = {"harbor_task": "/tmp/task", "harbor_agent_import_path": "x:y"}
        with pytest.raises(ValueError, match="cannot receive the native-harbor"):
            backend._build_agent_config(_request(md), _ctx(), "SID")

    def test_installed_agent_rejected_when_identity_headers_required(self):
        # Installed CLIs cannot forward x-rollout-id/x-sample-id. With the
        # default inject_identity_headers=True the controller requires them, so
        # the backend must fail loud instead of silently 400-ing every call.
        backend = NativeHarborBackend()
        with pytest.raises(ValueError, match="not yet supported"):
            backend._build_agent_config(
                _request({"harbor_task": "/tmp/task", "harbor_agent": "codex"}),
                _ctx(),
                "SID",
            )

    def test_installed_agent_uses_env_transport_when_headers_off(self):
        # With inject_identity_headers=False the controller routes identity
        # another way, so installed agents are wired via env vars only -- the
        # endpoint goes through harbor's standard env channel, not llm_kwargs.
        backend = NativeHarborBackend(inject_identity_headers=False)
        ac = backend._build_agent_config(
            _request({"harbor_task": "/tmp/task", "harbor_agent": "codex"}),
            _ctx(),
            "SID",
        )
        assert ac.env == {
            "OPENAI_BASE_URL": "http://ctrl:8080",
            "OPENAI_API_KEY": "sk-test",
        }
        assert ac.kwargs == {}  # no in-process knobs for installed agents

    def test_metadata_overrides_agent_and_model(self):
        backend = NativeHarborBackend()
        md = {
            "harbor_task": "/tmp/task",
            "harbor_agent_import_path": "my.pkg:MyAgent",
            "harbor_model": "openai/custom",
        }
        ac = backend._build_agent_config(_request(md), _ctx(), "SID")
        assert ac.name is None
        assert ac.import_path == "my.pkg:MyAgent"
        assert ac.model_name == "openai/custom"
        # Unresolvable import_path -> class is None -> no knobs forced; the
        # canonical "unknown agent" error is left to Trial.create.
        assert "enable_summarize" not in ac.kwargs

    def test_agent_timeout_forwarded(self):
        backend = NativeHarborBackend()
        ac = backend._build_agent_config(
            _request(agent_timeout_sec=42.0), _ctx(), "SID"
        )
        assert ac.override_timeout_sec == 42.0

    def test_both_agent_selectors_raise(self):
        backend = NativeHarborBackend()
        md = {
            "harbor_task": "/tmp/task",
            "harbor_agent": "terminus-2",
            "harbor_agent_import_path": "my.pkg:MyAgent",
        }
        with pytest.raises(ValueError, match="choose one"):
            backend._build_agent_config(_request(md), _ctx(), "SID")

    def test_blank_agent_falls_back_to_default(self):
        # A blank dataset cell deserializes to "" and must fall back to the
        # default agent, not build AgentConfig(name="") and fail the trial.
        backend = NativeHarborBackend()
        ac = backend._build_agent_config(
            _request({"harbor_task": "/tmp/task", "harbor_agent": ""}), _ctx(), "SID"
        )
        assert ac.name == "terminus-2"


class TestRewardPicking:
    def test_named_channel(self):
        assert NativeHarborBackend()._pick_reward({"reward": 1}) == 1

    def test_sole_value_fallback(self):
        assert NativeHarborBackend()._pick_reward({"accuracy": 0.8}) == 0.8

    def test_ambiguous_returns_none(self):
        backend = NativeHarborBackend()
        assert backend._pick_reward({"a": 1, "b": 2}) is None

    def test_custom_reward_key(self):
        backend = NativeHarborBackend(reward_key="score")
        assert backend._pick_reward({"score": 0.3, "reward": 0.9}) == 0.3

    def test_extract_top_level(self):
        assert NativeHarborBackend()._extract_rewards(
            _trial_result(rewards={"reward": 1.0})
        ) == {"reward": 1.0}

    def test_extract_multi_step_fallback(self):
        assert NativeHarborBackend()._extract_rewards(
            _trial_result(steps=[{"reward": 0.5}])
        ) == {"reward": 0.5}


class TestExecute:
    def test_is_execution_backend(self):
        assert isinstance(NativeHarborBackend(), ExecutionBackend)

    async def test_success_sample_id_round_trip(self, monkeypatch):
        capture: dict[str, Any] = {}
        _patch_trial(
            monkeypatch, result=_trial_result(rewards={"reward": 1.0}), capture=capture
        )
        backend = NativeHarborBackend()
        on_wf, on_gr = AsyncMock(), AsyncMock()

        with _ctx():
            await backend.execute(_request(), on_wf, on_gr)

        on_wf.assert_awaited_once()
        on_gr.assert_awaited_once()
        wf_result = on_wf.call_args.args[0]
        gr_result = on_gr.call_args.args[0]
        assert wf_result.status == RolloutStatus.SUCCESS
        assert gr_result.status == RolloutStatus.SUCCESS

        injected_sid = capture["config"].agent.kwargs["llm_kwargs"]["extra_headers"][
            "x-sample-id"
        ]
        # THE invariant: injected x-sample-id == samples dict key == reward sample.
        assert list(gr_result.samples.keys()) == [injected_sid]
        assert gr_result.samples[injected_sid].reward == 1.0
        assert (
            capture["config"].agent.kwargs["llm_kwargs"]["extra_headers"][
                "x-rollout-id"
            ]
            == "ROLL"
        )
        # verifier must be enabled so it produces a reward.
        assert capture["config"].verifier.disable is False

    async def test_int_reward_coerced_to_float(self, monkeypatch):
        _patch_trial(monkeypatch, result=_trial_result(rewards={"reward": 1}))
        backend = NativeHarborBackend()
        on_wf, on_gr = AsyncMock(), AsyncMock()
        with _ctx():
            await backend.execute(_request(), on_wf, on_gr)
        sample = next(iter(on_gr.call_args.args[0].samples.values()))
        assert sample.reward == 1.0
        assert isinstance(sample.reward, float)

    async def test_agent_failure_fires_both_callbacks(self, monkeypatch):
        _patch_trial(monkeypatch, result=_trial_result(exc_message="boom"))
        backend = NativeHarborBackend()
        on_wf, on_gr = AsyncMock(), AsyncMock()
        with _ctx():
            await backend.execute(_request(), on_wf, on_gr)
        wf_result = on_wf.call_args.args[0]
        gr_result = on_gr.call_args.args[0]
        assert wf_result.status == RolloutStatus.FAILURE
        assert wf_result.err_message == "boom"
        assert wf_result.err_category == RolloutErrorCategory.AGENT_ERROR
        # grader fires FAILURE with the still-unrewarded sample (training fast-fails).
        assert gr_result.status == RolloutStatus.FAILURE
        assert next(iter(gr_result.samples.values())).reward is None

    async def test_trial_create_raises(self, monkeypatch):
        _patch_trial(monkeypatch, create_error=RuntimeError("docker down"))
        backend = NativeHarborBackend()
        on_wf, on_gr = AsyncMock(), AsyncMock()
        with _ctx():
            await backend.execute(_request(), on_wf, on_gr)
        assert on_wf.call_args.args[0].status == RolloutStatus.FAILURE
        assert "docker down" in on_wf.call_args.args[0].err_message
        assert on_gr.call_args.args[0].status == RolloutStatus.FAILURE

    async def test_missing_task_is_validation_error(self, monkeypatch):
        _patch_trial(monkeypatch)
        backend = NativeHarborBackend()
        on_wf, on_gr = AsyncMock(), AsyncMock()
        with _ctx():
            await backend.execute(_request({}), on_wf, on_gr)
        assert (
            on_wf.call_args.args[0].err_category
            == RolloutErrorCategory.VALIDATION_ERROR
        )

    async def test_no_grader_callback_only_workflow(self, monkeypatch):
        _patch_trial(monkeypatch, result=_trial_result(rewards={"reward": 1.0}))
        backend = NativeHarborBackend()
        on_wf = AsyncMock()
        with _ctx():
            await backend.execute(_request(), on_wf, None)
        on_wf.assert_awaited_once()

    async def test_empty_rewards_is_validation_failure(self, monkeypatch):
        _patch_trial(monkeypatch, result=_trial_result(rewards={}))
        backend = NativeHarborBackend()
        on_wf, on_gr = AsyncMock(), AsyncMock()
        with _ctx():
            await backend.execute(_request(), on_wf, on_gr)
        # workflow succeeded but grading found no reward -> VALIDATION_ERROR.
        assert on_wf.call_args.args[0].status == RolloutStatus.SUCCESS
        gr_result = on_gr.call_args.args[0]
        assert gr_result.status == RolloutStatus.FAILURE
        assert gr_result.err_category == RolloutErrorCategory.VALIDATION_ERROR

    async def test_both_agent_selectors_is_validation_error(self, monkeypatch):
        _patch_trial(monkeypatch)
        backend = NativeHarborBackend()
        on_wf, on_gr = AsyncMock(), AsyncMock()
        md = {
            "harbor_task": "/tmp/task",
            "harbor_agent": "terminus-2",
            "harbor_agent_import_path": "my.pkg:MyAgent",
        }
        with _ctx():
            await backend.execute(_request(md), on_wf, on_gr)
        assert (
            on_wf.call_args.args[0].err_category
            == RolloutErrorCategory.VALIDATION_ERROR
        )

    async def test_reward_salvaged_when_exception_after_verify(self, monkeypatch):
        # exception_info set AND a reward present: the verifier already computed a
        # reward, so it must not be discarded.
        _patch_trial(
            monkeypatch,
            result=_trial_result(
                rewards={"reward": 0.7}, exc_message="post-verify upload failed"
            ),
        )
        backend = NativeHarborBackend()
        on_wf, on_gr = AsyncMock(), AsyncMock()
        with _ctx():
            await backend.execute(_request(), on_wf, on_gr)
        # workflow honestly reflects the exception...
        assert on_wf.call_args.args[0].status == RolloutStatus.FAILURE
        # ...but grading salvages the real reward.
        gr_result = on_gr.call_args.args[0]
        assert gr_result.status == RolloutStatus.SUCCESS
        assert next(iter(gr_result.samples.values())).reward == 0.7

    async def test_exception_type_drives_err_category(self, monkeypatch):
        # A verifier timeout swallowed into exception_info must be reported as
        # TIMEOUT, not the old hard-coded AGENT_ERROR.
        _patch_trial(
            monkeypatch,
            result=_trial_result(
                exc_message="verifier timed out", exc_type="VerifierTimeoutError"
            ),
        )
        backend = NativeHarborBackend()
        on_wf, on_gr = AsyncMock(), AsyncMock()
        with _ctx():
            await backend.execute(_request(), on_wf, on_gr)
        assert on_wf.call_args.args[0].err_category == RolloutErrorCategory.TIMEOUT

    async def test_grader_callback_failure_does_not_propagate(self, monkeypatch):
        # A failing grader callback must NOT escape execute(): app.py's fallback
        # would otherwise re-fire both callbacks and break fire-exactly-once.
        _patch_trial(monkeypatch, result=_trial_result(rewards={"reward": 1.0}))
        backend = NativeHarborBackend()
        on_wf = AsyncMock()
        on_gr = AsyncMock(side_effect=RuntimeError("controller down"))
        with _ctx():
            await backend.execute(_request(), on_wf, on_gr)  # must not raise
        on_wf.assert_awaited_once()
        on_gr.assert_awaited_once()


class TestConcurrencyAndLifecycle:
    def test_unbounded_concurrency_rejected(self):
        with pytest.raises(ValueError, match="max_concurrent must be >= 1"):
            NativeHarborBackend(max_concurrent=0)

    def test_health_reports_max_concurrency(self):
        backend = NativeHarborBackend(max_concurrent=3)
        assert backend.max_concurrency == 3
        assert backend.health()["max_concurrency"] == 3

    async def test_successful_trial_dir_cleaned_up(self, monkeypatch, tmp_path):
        # On success the (disposable) trial dir is removed; the reward is read
        # from the in-memory TrialResult, not the files.
        _patch_trial(monkeypatch, result=_trial_result(rewards={"reward": 1.0}))
        backend = NativeHarborBackend(trials_dir=tmp_path)
        trial_dir = tmp_path / "native-ROLL-SID"
        trial_dir.mkdir()
        monkeypatch.setattr(bmod.uuid, "uuid4", lambda: SimpleNamespace(hex="SID"))
        with _ctx():
            await backend.execute(_request(), AsyncMock(), AsyncMock())
        assert not trial_dir.exists()

    async def test_failed_trial_dir_kept(self, monkeypatch, tmp_path):
        # Failed trials are kept for debugging.
        _patch_trial(monkeypatch, result=_trial_result(exc_message="boom"))
        backend = NativeHarborBackend(trials_dir=tmp_path)
        trial_dir = tmp_path / "native-ROLL-SID"
        trial_dir.mkdir()
        monkeypatch.setattr(bmod.uuid, "uuid4", lambda: SimpleNamespace(hex="SID"))
        with _ctx():
            await backend.execute(_request(), AsyncMock(), AsyncMock())
        assert trial_dir.exists()

    async def test_environment_config_threaded_into_trial(self, monkeypatch):
        # The sandbox type is trial-layer, so a caller-supplied EnvironmentConfig
        # (e.g. daytona) must reach TrialConfig.environment.
        from harbor.models.environment_type import EnvironmentType
        from harbor.models.trial.config import EnvironmentConfig

        capture: dict[str, Any] = {}
        _patch_trial(
            monkeypatch, result=_trial_result(rewards={"reward": 1.0}), capture=capture
        )
        backend = NativeHarborBackend(
            environment_config=EnvironmentConfig(type=EnvironmentType.DAYTONA)
        )
        with _ctx():
            await backend.execute(_request(), AsyncMock(), AsyncMock())
        assert capture["config"].environment.type == EnvironmentType.DAYTONA
