"""Unit tests for the native harbor backend.

The backend drives a real harbor ``Trial`` per rollout, so these tests exercise
the pure seams around it: task resolution, agent wiring, and the mapping from a
harbor ``TrialResult`` onto the rollout's single sample. The trial itself is
stubbed at ``TrialQueue.submit``.
"""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import EnvironmentConfig as HarborEnvironmentConfig

from osmosis_ai.rollout.backend.native_harbor.backend import (
    DEFAULT_AGENT_NAME,
    NativeHarborBackend,
    _is_installed_agent,
    _resolve_agent_class,
    resolve_task,
)
from osmosis_ai.rollout.context import RolloutContext
from osmosis_ai.rollout.types import (
    ExecutionRequest,
    ExecutionResult,
    RolloutErrorCategory,
    RolloutStatus,
)

INSTALLED_AGENT_NAME = "claude-code"


def _request(**overrides: Any) -> ExecutionRequest:
    payload: dict[str, Any] = {
        "id": "rollout-1",
        "prompt": [{"role": "user", "content": "hi"}],
        "metadata": {"harbor_task": "osmosis/demo"},
    }
    payload.update(overrides)
    return ExecutionRequest(**payload)


def _context(**overrides: Any) -> RolloutContext:
    payload: dict[str, Any] = {
        "chat_completions_url": "http://localhost:8080/v1/chat/completions",
        "api_key": "sk-test",
        "rollout_id": "rollout-1",
    }
    payload.update(overrides)
    return RolloutContext(**payload)


def _trial_result(
    *,
    rewards: dict[str, float] | None = None,
    step_rewards: dict[str, float] | None = None,
    exception_info: Any = None,
) -> SimpleNamespace:
    """A duck-typed harbor ``TrialResult``: only these attributes are read."""
    return SimpleNamespace(
        verifier_result=SimpleNamespace(rewards=rewards) if rewards else None,
        step_results=(
            [SimpleNamespace(verifier_result=SimpleNamespace(rewards=step_rewards))]
            if step_rewards
            else []
        ),
        exception_info=exception_info,
    )


class TestResolveTask:
    def test_local_path_expands_user(self):
        cfg = resolve_task(_request(metadata={"harbor_task": "~/tasks/demo"}))

        assert cfg.path == Path("~/tasks/demo").expanduser()

    @pytest.mark.parametrize("prefix", ["./", "/"])
    def test_local_path_prefixes(self, prefix):
        cfg = resolve_task(_request(metadata={"harbor_task": f"{prefix}tasks/demo"}))

        assert cfg.path == Path(f"{prefix}tasks/demo")

    def test_package_defaults_to_latest_ref(self):
        cfg = resolve_task(_request(metadata={"harbor_task": "osmosis/demo"}))

        assert (cfg.name, cfg.ref) == ("osmosis/demo", "latest")

    def test_package_honors_explicit_ref(self):
        cfg = resolve_task(_request(metadata={"harbor_task": "osmosis/demo@v2"}))

        assert (cfg.name, cfg.ref) == ("osmosis/demo", "v2")

    def test_git_form_takes_precedence_over_package_parsing(self):
        cfg = resolve_task(
            _request(
                metadata={
                    "harbor_task": "ignored",
                    "git_url": "https://example.test/tasks.git",
                    "task_path": "tasks/demo",
                    "git_commit_id": "abc123",
                }
            )
        )

        assert cfg.git_url == "https://example.test/tasks.git"
        assert cfg.path == Path("tasks/demo")
        assert cfg.git_commit_id == "abc123"

    def test_missing_task_is_rejected(self):
        with pytest.raises(ValueError, match="harbor_task"):
            resolve_task(_request(metadata={}))

    def test_bare_name_without_org_is_rejected(self):
        with pytest.raises(ValueError, match="package"):
            resolve_task(_request(metadata={"harbor_task": "demo"}))


class TestResolveAgentClass:
    """Unresolvable agents return None so Trial.create raises harbor's own error
    rather than a lookalike from this backend."""

    def test_resolves_a_registered_agent_name(self):
        cls = _resolve_agent_class(DEFAULT_AGENT_NAME, None)

        assert cls is not None
        assert _is_installed_agent(cls) is False

    def test_resolves_an_installed_agent_name(self):
        assert _is_installed_agent(_resolve_agent_class(INSTALLED_AGENT_NAME, None))

    def test_resolves_an_import_path(self):
        cls = _resolve_agent_class(
            None,
            "osmosis_ai.rollout.backend.harbor.agent_adapter:OsmosisInstalledAgent",
        )

        assert _is_installed_agent(cls)

    @pytest.mark.parametrize(
        "import_path",
        [
            "missing_colon",
            "osmosis_ai.does_not_exist:Agent",
            "osmosis_ai.rollout.backend.native_harbor.backend:NoSuchAgent",
        ],
    )
    def test_unresolvable_import_paths_yield_none(self, import_path):
        assert _resolve_agent_class(None, import_path) is None

    def test_unknown_agent_name_yields_none(self):
        assert _resolve_agent_class("no-such-agent", None) is None

    def test_none_is_not_an_installed_agent(self):
        assert _is_installed_agent(None) is False


class TestInit:
    def test_rejects_zero_concurrency(self):
        with pytest.raises(ValueError, match="max_concurrent"):
            NativeHarborBackend(max_concurrent=0)

    def test_rejects_both_agent_name_and_import_path(self):
        with pytest.raises(ValueError, match="not both"):
            NativeHarborBackend(agent_name="a", agent_import_path="m:C")

    def test_defaults_to_terminus_2(self):
        backend = NativeHarborBackend()

        assert backend.agent_name == DEFAULT_AGENT_NAME
        assert backend.max_concurrency == 8
        assert backend.health()["backend"] == "native_harbor"

    def test_import_path_leaves_agent_name_unset(self):
        backend = NativeHarborBackend(agent_import_path="pkg.mod:Agent")

        assert backend.agent_name is None
        assert backend.health()["agent"] == "pkg.mod:Agent"


class TestManagedSkypilotPlacement:
    """Native trials resolve placement the same way HarborBackend does, so a
    rollout can select SkyPilot without naming a cluster."""

    def test_fills_context_name_from_environment(self, monkeypatch):
        monkeypatch.setenv("HARBOR_SKYPILOT_CONTEXT", "managed-context")

        backend = NativeHarborBackend(
            environment_config=HarborEnvironmentConfig(type=EnvironmentType.SKYPILOT)
        )

        assert backend.environment_config.kwargs["context_name"] == "managed-context"

    def test_explicit_context_name_wins(self, monkeypatch):
        monkeypatch.setenv("HARBOR_SKYPILOT_CONTEXT", "managed-context")

        backend = NativeHarborBackend(
            environment_config=HarborEnvironmentConfig(
                type=EnvironmentType.SKYPILOT, kwargs={"context_name": "mine"}
            )
        )

        assert backend.environment_config.kwargs["context_name"] == "mine"

    def test_docker_default_is_untouched(self, monkeypatch):
        monkeypatch.setenv("HARBOR_SKYPILOT_CONTEXT", "managed-context")

        backend = NativeHarborBackend()

        assert backend.environment_config.type == EnvironmentType.DOCKER
        assert backend.environment_config.kwargs == {}


class TestInProcessAgentConfig:
    def test_missing_endpoint_is_rejected(self):
        backend = NativeHarborBackend()

        with pytest.raises(ValueError, match="chat_completions_url"):
            backend._build_agent_config(_request(), _context(chat_completions_url=""))

    def test_identity_headers_overwrite_caller_values(self):
        backend = NativeHarborBackend(
            agent_kwargs={
                "llm_kwargs": {
                    "extra_headers": {
                        "x-rollout-id": "someone-else",
                        "x-sample-id": "someone-else",
                        "x-trace": "keep-me",
                    }
                }
            }
        )

        cfg = backend._build_agent_config(_request(), _context())

        headers = cfg.kwargs["llm_kwargs"]["extra_headers"]
        assert headers["x-rollout-id"] == "rollout-1"
        assert headers["x-sample-id"] == "rollout-1"
        assert headers["x-trace"] == "keep-me"

    def test_endpoint_is_not_rewritten_for_the_host_process(self, monkeypatch):
        # An in-process agent runs in this process, not in the container, so a
        # loopback endpoint reaches the bridge as-is even on macOS Docker.
        monkeypatch.setattr(
            "osmosis_ai.rollout.backend.harbor.backend.platform.system",
            lambda: "Darwin",
        )
        backend = NativeHarborBackend()

        cfg = backend._build_agent_config(_request(), _context())

        assert cfg.kwargs["api_base"] == "http://localhost:8080/v1/chat/completions"

    def test_sdk_wiring_overrides_caller_endpoint_and_key(self):
        backend = NativeHarborBackend(
            agent_kwargs={
                "api_base": "http://evil.test",
                "llm_kwargs": {"api_key": "sk-caller"},
            }
        )

        cfg = backend._build_agent_config(_request(), _context())

        assert cfg.kwargs["api_base"] == "http://localhost:8080/v1/chat/completions"
        assert cfg.kwargs["llm_kwargs"]["api_key"] == "sk-test"

    def test_terminus_2_summarization_defaults_are_applied(self):
        backend = NativeHarborBackend()

        cfg = backend._build_agent_config(_request(), _context())

        assert cfg.kwargs["enable_summarize"] is False
        assert cfg.kwargs["proactive_summarization_threshold"] == 0

    def test_caller_kwargs_override_the_default_agent_kwargs(self):
        backend = NativeHarborBackend(agent_kwargs={"enable_summarize": True})

        cfg = backend._build_agent_config(_request(), _context())

        assert cfg.kwargs["enable_summarize"] is True

    def test_streaming_is_pinned_off_unless_the_caller_sets_it(self):
        assert (
            NativeHarborBackend()
            ._build_agent_config(_request(), _context())
            .kwargs["llm_kwargs"]["extra_body"]["stream"]
            is False
        )
        assert (
            NativeHarborBackend(
                agent_kwargs={"llm_kwargs": {"extra_body": {"stream": True}}}
            )
            ._build_agent_config(_request(), _context())
            .kwargs["llm_kwargs"]["extra_body"]["stream"]
            is True
        )

    def test_model_name_comes_from_metadata_when_present(self):
        backend = NativeHarborBackend()

        cfg = backend._build_agent_config(
            _request(metadata={"harbor_task": "osmosis/demo", "harbor_model": "x/y"}),
            _context(),
        )

        assert cfg.model_name == "x/y"

    def test_agent_timeout_is_forwarded(self):
        backend = NativeHarborBackend()

        cfg = backend._build_agent_config(_request(agent_timeout_sec=30.0), _context())

        assert cfg.override_timeout_sec == 30.0


class TestInstalledAgentConfig:
    """An installed agent runs inside the environment container, so its endpoint
    has to survive the container's network namespace."""

    def _backend(self, **kwargs: Any) -> NativeHarborBackend:
        return NativeHarborBackend(agent_name=INSTALLED_AGENT_NAME, **kwargs)

    def test_loopback_is_rewritten_on_macos_docker(self, monkeypatch):
        monkeypatch.setattr(
            "osmosis_ai.rollout.backend.harbor.backend.platform.system",
            lambda: "Darwin",
        )

        cfg = self._backend()._build_agent_config(_request(), _context())

        assert (
            cfg.env["OPENAI_BASE_URL"]
            == "http://host.docker.internal:8080/v1/chat/completions"
        )
        assert cfg.env["OPENAI_API_KEY"] == "sk-test"

    def test_endpoint_is_untouched_on_linux_docker(self, monkeypatch):
        monkeypatch.setattr(
            "osmosis_ai.rollout.backend.harbor.backend.platform.system",
            lambda: "Linux",
        )

        cfg = self._backend()._build_agent_config(_request(), _context())

        assert cfg.env["OPENAI_BASE_URL"] == "http://localhost:8080/v1/chat/completions"

    def test_endpoint_is_untouched_for_remote_environments(self, monkeypatch):
        monkeypatch.setattr(
            "osmosis_ai.rollout.backend.harbor.backend.platform.system",
            lambda: "Darwin",
        )
        backend = self._backend(
            environment_config=HarborEnvironmentConfig(type=EnvironmentType.DAYTONA)
        )

        cfg = backend._build_agent_config(_request(), _context())

        assert cfg.env["OPENAI_BASE_URL"] == "http://localhost:8080/v1/chat/completions"

    def test_sdk_wiring_overwrites_caller_env(self, monkeypatch):
        monkeypatch.setattr(
            "osmosis_ai.rollout.backend.harbor.backend.platform.system",
            lambda: "Linux",
        )
        backend = self._backend(
            agent_env={"OPENAI_BASE_URL": "http://evil.test", "KEEP": "yes"}
        )

        cfg = backend._build_agent_config(_request(), _context())

        assert cfg.env["OPENAI_BASE_URL"] == "http://localhost:8080/v1/chat/completions"
        assert cfg.env["KEEP"] == "yes"
        assert cfg.kwargs == {}


class TestTrialConfig:
    @staticmethod
    def _build(backend: NativeHarborBackend, request: ExecutionRequest):
        return backend._build_trial_config(
            request,
            resolve_task(request),
            backend._build_agent_config(request, _context()),
            "native-rollout-1",
        )

    def test_grader_timeout_is_forwarded_to_the_verifier(self):
        cfg = self._build(NativeHarborBackend(), _request(grader_timeout_sec=12.0))

        assert cfg.verifier.override_timeout_sec == 12.0
        assert cfg.verifier.disable is False

    def test_verifier_runs_untimed_by_default(self):
        cfg = self._build(NativeHarborBackend(), _request())

        assert cfg.verifier.override_timeout_sec is None
        assert cfg.verifier.disable is False

    def test_paths_and_identity_are_threaded_through(self):
        backend = NativeHarborBackend(trials_dir="native_trials")

        cfg = self._build(backend, _request())

        assert cfg.trials_dir == Path("native_trials")
        assert cfg.task.name == "osmosis/demo"
        assert cfg.trial_name == "native-rollout-1"
        assert cfg.environment is backend.environment_config


class TestRewardExtraction:
    def test_prefers_the_trial_level_reward(self):
        rewards = NativeHarborBackend._extract_rewards(
            _trial_result(rewards={"reward": 1.0}, step_rewards={"reward": 0.0})
        )

        assert rewards == {"reward": 1.0}

    def test_falls_back_to_the_first_step_with_rewards(self):
        rewards = NativeHarborBackend._extract_rewards(
            _trial_result(step_rewards={"reward": 0.5})
        )

        assert rewards == {"reward": 0.5}

    def test_returns_none_without_any_rewards(self):
        assert NativeHarborBackend._extract_rewards(_trial_result()) is None

    def test_picks_the_configured_reward_key(self):
        assert NativeHarborBackend()._pick_reward({"reward": 1.0, "other": 0.0}) == 1.0

    def test_picks_a_sole_unnamed_channel(self):
        assert NativeHarborBackend()._pick_reward({"pass_rate": 0.25}) == 0.25

    def test_honors_a_custom_reward_key(self):
        backend = NativeHarborBackend(reward_key="pass_rate")

        assert backend._pick_reward({"pass_rate": 0.25, "reward": 1.0}) == 0.25

    def test_ambiguous_channels_leave_the_reward_unset(self, caplog):
        with caplog.at_level("WARNING"):
            assert NativeHarborBackend()._pick_reward({"a": 1.0, "b": 0.0}) is None

        assert "no 'reward' channel" in caplog.text

    def test_empty_rewards_pick_nothing(self):
        assert NativeHarborBackend()._pick_reward(None) is None
        assert NativeHarborBackend()._pick_reward({}) is None


class TestGraderResult:
    def _grade(
        self, workflow_result: ExecutionResult, trial_result: Any
    ) -> ExecutionResult:
        return NativeHarborBackend()._build_grader_result(
            _request(label="demo"), workflow_result, trial_result
        )

    def test_reward_maps_onto_the_single_sample(self):
        result = self._grade(
            ExecutionResult(status=RolloutStatus.SUCCESS),
            _trial_result(rewards={"reward": 1.0}),
        )

        assert result.status == RolloutStatus.SUCCESS
        assert result.samples["rollout-1"].reward == 1.0
        assert result.samples["rollout-1"].label == "demo"

    def test_setup_failure_propagates_the_workflow_error(self):
        result = self._grade(
            ExecutionResult(
                status=RolloutStatus.FAILURE,
                err_message="task not found",
                err_category=RolloutErrorCategory.VALIDATION_ERROR,
            ),
            None,
        )

        assert result.err_message == "task not found"
        assert result.err_category == RolloutErrorCategory.VALIDATION_ERROR

    def test_in_trial_failure_propagates_the_agent_error(self):
        # A failed trial still carries a TrialResult, but no verifier reward. The
        # agent failure is the diagnosis, not the missing reward it caused.
        result = self._grade(
            ExecutionResult(
                status=RolloutStatus.FAILURE,
                err_message="agent crashed",
                err_category=RolloutErrorCategory.AGENT_ERROR,
            ),
            _trial_result(),
        )

        assert result.status == RolloutStatus.FAILURE
        assert result.err_message == "agent crashed"
        assert result.err_category == RolloutErrorCategory.AGENT_ERROR

    def test_successful_trial_without_rewards_is_a_validation_error(self):
        result = self._grade(
            ExecutionResult(status=RolloutStatus.SUCCESS), _trial_result()
        )

        assert result.status == RolloutStatus.FAILURE
        assert result.err_category == RolloutErrorCategory.VALIDATION_ERROR


class TestExecute:
    """``execute`` fires each callback exactly once and never propagates."""

    def _backend(self, monkeypatch, submit: Any) -> NativeHarborBackend:
        backend = NativeHarborBackend(cleanup_successful_trials=False)
        monkeypatch.setattr(backend._queue, "submit", submit)
        return backend

    async def _run(self, backend: NativeHarborBackend) -> tuple[list, list]:
        workflow: list[ExecutionResult] = []
        grader: list[ExecutionResult] = []

        async def on_workflow(result):
            workflow.append(result)

        async def on_grader(result):
            grader.append(result)

        with _context():
            await backend.execute(_request(), on_workflow, on_grader)
        return workflow, grader

    @pytest.mark.asyncio
    async def test_successful_trial_reports_reward(self, monkeypatch):
        async def submit(_cfg):
            return _trial_result(rewards={"reward": 1.0})

        workflow, grader = await self._run(self._backend(monkeypatch, submit))

        assert workflow[0].status == RolloutStatus.SUCCESS
        assert grader[0].samples["rollout-1"].reward == 1.0

    @pytest.mark.asyncio
    async def test_setup_failure_reaches_both_callbacks(self, monkeypatch):
        async def submit(_cfg):
            raise TimeoutError("trial timed out")

        workflow, grader = await self._run(self._backend(monkeypatch, submit))

        assert workflow[0].err_message == "trial timed out"
        assert grader[0].err_message == "trial timed out"

    @pytest.mark.parametrize(
        ("exc", "category"),
        [
            (TimeoutError("slow"), RolloutErrorCategory.TIMEOUT),
            (ValueError("bad task"), RolloutErrorCategory.VALIDATION_ERROR),
            (RuntimeError("boom"), RolloutErrorCategory.AGENT_ERROR),
        ],
    )
    @pytest.mark.asyncio
    async def test_setup_failures_are_categorized(self, monkeypatch, exc, category):
        async def submit(_cfg):
            raise exc

        workflow, _ = await self._run(self._backend(monkeypatch, submit))

        assert workflow[0].err_category == category

    @pytest.mark.asyncio
    async def test_in_trial_exception_surfaces_the_harbor_message(self, monkeypatch):
        async def submit(_cfg):
            return _trial_result(
                exception_info=SimpleNamespace(
                    exception_type="RuntimeError",
                    exception_message="agent crashed",
                )
            )

        workflow, grader = await self._run(self._backend(monkeypatch, submit))

        assert workflow[0].err_message == "agent crashed"
        assert grader[0].err_category == RolloutErrorCategory.AGENT_ERROR

    @pytest.mark.asyncio
    async def test_grader_callback_is_skipped_when_absent(self, monkeypatch):
        async def submit(_cfg):
            return _trial_result(rewards={"reward": 1.0})

        backend = self._backend(monkeypatch, submit)
        seen: list[ExecutionResult] = []

        async def on_workflow(result):
            seen.append(result)

        with _context():
            await backend.execute(_request(), on_workflow)

        assert len(seen) == 1

    @pytest.mark.asyncio
    async def test_callback_failure_does_not_propagate(self, monkeypatch):
        async def submit(_cfg):
            return _trial_result(rewards={"reward": 1.0})

        backend = self._backend(monkeypatch, submit)
        graded: list[ExecutionResult] = []

        async def on_workflow(_result):
            raise RuntimeError("callback exploded")

        async def on_grader(result):
            graded.append(result)

        with _context():
            await backend.execute(_request(), on_workflow, on_grader)

        # A propagating workflow callback would re-fire both in app.py.
        assert len(graded) == 1


class TestCleanup:
    def test_successful_trial_dir_is_removed(self, tmp_path):
        backend = NativeHarborBackend(trials_dir=tmp_path)
        trial_dir = tmp_path / "native-rollout-1"
        trial_dir.mkdir()

        backend._cleanup_trial("native-rollout-1")

        assert not trial_dir.exists()

    def test_cleanup_can_be_disabled(self, tmp_path):
        backend = NativeHarborBackend(
            trials_dir=tmp_path, cleanup_successful_trials=False
        )
        trial_dir = tmp_path / "native-rollout-1"
        trial_dir.mkdir()

        backend._cleanup_trial("native-rollout-1")

        assert trial_dir.exists()
