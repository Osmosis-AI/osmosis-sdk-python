"""Unit tests for the NativeHarborBackend.

The harbor ``Trial`` is monkeypatched so these run without Docker/harbor task
resolution. A rollout produces a single, URL-routed sample: identity is baked
into ``chat_completions_url`` (and the callback URLs), so the backend no longer
stamps per-call ``x-rollout-id``/``x-sample-id`` headers, and the verifier
reward lands on the one ``ExecutionResult.sample``.
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
    def test_in_process_url_and_endpoint_wiring(self):
        # The in-process default agent gets the endpoint/key wired via kwargs.
        # Identity rides in the URL path, so no per-call routing headers are set.
        # (Kept independent of harbor's agent registry: the summarization-knob
        # forcing depends on class resolution and is covered separately.)
        backend = NativeHarborBackend()
        ac = backend._build_agent_config(_request(), _ctx())
        assert ac.name == "terminus-2"
        # api_base is top-level; api_key rides inside llm_kwargs.
        assert ac.kwargs["api_base"] == "http://ctrl:8080"
        assert ac.kwargs["collect_rollout_details"] is False
        assert ac.kwargs["llm_kwargs"]["api_key"] == "sk-test"
        # Rollout identity rides in the URL path now -- no per-call routing headers.
        assert "extra_headers" not in ac.kwargs["llm_kwargs"]
        # The non-streaming hack is unrelated to URL routing and stays forced.
        assert ac.kwargs["llm_kwargs"]["extra_body"] == {"stream": False}

    def test_training_safe_off_omits_summarize_knobs(self):
        backend = NativeHarborBackend(training_safe=False)
        ac = backend._build_agent_config(_request(), _ctx())
        assert "enable_summarize" not in ac.kwargs

    def test_unwhitelisted_builtin_agent_raises_under_training_safe(self):
        # A harbor built-in that is not on the training-safe whitelist (here nop)
        # is rejected, not silently run: harbor 0.15 cannot guarantee its
        # trajectory stays linear, and a silent pass would corrupt training data.
        backend = NativeHarborBackend(agent_name="nop")  # training_safe=True default
        with pytest.raises(ValueError, match="training-safe whitelist"):
            backend._build_agent_config(_request(), _ctx())

    def test_unwhitelisted_builtin_agent_runs_in_eval_mode(self):
        # With training_safe=False (eval) the same built-in is left alone: no
        # summarize knobs forced, no rejection -- reward-only runs do not need a
        # linear token trajectory.
        backend = NativeHarborBackend(agent_name="nop", training_safe=False)
        ac = backend._build_agent_config(_request(), _ctx())
        assert "enable_summarize" not in ac.kwargs

    def test_summarize_knobs_follow_resolved_class_not_name(self):
        # import_path into harbor.* resolves to the class, so knobs are forced by
        # issubclass even though name is None.
        backend = NativeHarborBackend(
            agent_import_path="harbor.agents.terminus_2:Terminus2"
        )
        ac = backend._build_agent_config(_request(), _ctx())
        assert ac.name is None
        assert ac.import_path == "harbor.agents.terminus_2:Terminus2"
        assert ac.kwargs["enable_summarize"] is False
        assert ac.kwargs["proactive_summarization_threshold"] == 0

    def test_custom_agent_not_gated_and_not_injected(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        # A user agent (import_path outside harbor.*) is trusted: not rejected by
        # the gate, knobs not injected -- training safety is the author's job.
        class _CustomAgent:
            def __init__(self, logs_dir=None, enable_summarize=True, **kwargs):
                pass

        monkeypatch.setattr(bmod, "_resolve_agent_class", lambda name, ip: _CustomAgent)
        backend = NativeHarborBackend(agent_import_path="my.custom.pkg:CustomAgent")
        ac = backend._build_agent_config(_request(), _ctx())
        assert ac.import_path == "my.custom.pkg:CustomAgent"
        assert ac.name is None
        assert ac.kwargs["api_base"] == "http://ctrl:8080"  # wired, not rejected
        assert "enable_summarize" not in ac.kwargs  # not injected

    def test_agent_name_and_import_path_mutually_exclusive(self):
        with pytest.raises(ValueError, match="not both"):
            NativeHarborBackend(
                agent_name="terminus-2", agent_import_path="my.pkg:MyAgent"
            )

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
        backend = NativeHarborBackend(training_safe=False)
        with pytest.raises(ValueError, match="cannot receive the native-harbor"):
            backend._build_agent_config(_request(), _ctx())

    def test_installed_agent_wired_via_env_url(self, monkeypatch: pytest.MonkeyPatch):
        # Installed CLIs (codex, claude-code, ...) are wired via env in eval mode:
        # the rollout id is baked into the chat-completions URL path, so
        # OPENAI_BASE_URL alone carries routing identity -- no per-call headers,
        # which installed CLIs cannot send. training_safe=False because a CLI
        # agent is not training-safe (covered by the next test); this asserts the
        # eval-mode wiring. Force the installed-agent branch so the test does not
        # depend on which harbor version's agent registry happens to be installed.
        monkeypatch.setattr(bmod, "_is_installed_agent", lambda cls: True)
        backend = NativeHarborBackend(agent_name="codex", training_safe=False)
        ac = backend._build_agent_config(_request(), _ctx())
        assert ac.env == {
            "OPENAI_BASE_URL": "http://ctrl:8080",
            "OPENAI_API_KEY": "sk-test",
        }
        assert ac.kwargs == {}  # no in-process knobs for installed agents

    def test_installed_builtin_agent_raises_under_training_safe(self):
        # An installed CLI built-in (codex) manages context inside an opaque
        # external process -- not training-safe and not on the whitelist, so
        # training_safe rejects it up front, before the env-wiring branch
        # (independent of whether the agent class is importable here).
        backend = NativeHarborBackend(agent_name="codex")
        with pytest.raises(ValueError, match="training-safe whitelist"):
            backend._build_agent_config(_request(), _ctx())

    def test_metadata_overrides_model(self):
        # The agent is fixed per backend, but the model name can still be
        # overridden per rollout via metadata.
        backend = NativeHarborBackend()
        md = {"harbor_task": "/tmp/task", "harbor_model": "openai/custom"}
        ac = backend._build_agent_config(_request(md), _ctx())
        assert ac.name == "terminus-2"
        assert ac.model_name == "openai/custom"

    def test_agent_timeout_forwarded(self):
        backend = NativeHarborBackend()
        ac = backend._build_agent_config(_request(agent_timeout_sec=42.0), _ctx())
        assert ac.override_timeout_sec == 42.0


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

    async def test_success_single_sample_rewarded(self, monkeypatch):
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

        # Single-sample contract: the verifier reward lands on the one sample.
        assert gr_result.sample.reward == 1.0
        # Identity rides in the URL path -- no per-call routing headers injected.
        assert "extra_headers" not in capture["config"].agent.kwargs["llm_kwargs"]
        # verifier must be enabled so it produces a reward.
        assert capture["config"].verifier.disable is False

    async def test_int_reward_coerced_to_float(self, monkeypatch):
        _patch_trial(monkeypatch, result=_trial_result(rewards={"reward": 1}))
        backend = NativeHarborBackend()
        on_wf, on_gr = AsyncMock(), AsyncMock()
        with _ctx():
            await backend.execute(_request(), on_wf, on_gr)
        sample = on_gr.call_args.args[0].sample
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
        assert gr_result.sample.reward is None

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
        assert gr_result.sample.reward == 0.7

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
        # from the in-memory TrialResult, not the files. The dir is named by the
        # rollout id alone (no per-execution token).
        _patch_trial(monkeypatch, result=_trial_result(rewards={"reward": 1.0}))
        backend = NativeHarborBackend(trials_dir=tmp_path)
        trial_dir = tmp_path / "native-ROLL"
        trial_dir.mkdir()
        with _ctx():
            await backend.execute(_request(), AsyncMock(), AsyncMock())
        assert not trial_dir.exists()

    async def test_failed_trial_dir_kept(self, monkeypatch, tmp_path):
        # Failed trials are kept for debugging.
        _patch_trial(monkeypatch, result=_trial_result(exc_message="boom"))
        backend = NativeHarborBackend(trials_dir=tmp_path)
        trial_dir = tmp_path / "native-ROLL"
        trial_dir.mkdir()
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
