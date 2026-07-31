"""Unit tests for ``NativeHarborBackend``.

Harbor queue submission is replaced with in-process trial results so configuration, callback, reward, and cleanup behavior can be tested without Docker.
"""

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import MCPServerConfig
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    VerifierConfig,
)

from osmosis_ai.rollout.backend.base import ExecutionBackend
from osmosis_ai.rollout.backend.native_harbor import backend as bmod
from osmosis_ai.rollout.backend.native_harbor.backend import (
    _AGENT_BINDINGS,
    NativeHarborBackend,
    _AgentProtocol,
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
    """Build a duck-typed Harbor trial result."""
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
    """Replace queue submission with an in-process fake.

    The fake optionally captures the generated TrialConfig and returns a duck-typed Harbor result.
    """

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


def _native_trajectory(*, api_key: str = "sk-test") -> dict[str, Any]:
    return {
        "schema_version": "ATIF-v1.7",
        "session_id": "native-session",
        "trajectory_id": "native-trajectory",
        "agent": {
            "name": "terminus-2",
            "version": "0.20.0",
            "model_name": "model",
            "extra": {
                "llm_kwargs": {
                    "api_key": api_key,
                    "temperature": 0,
                },
                "command": ["agent", "--token", api_key],
                "safe": "kept",
            },
        },
        "steps": [
            {
                "step_id": 1,
                "source": "agent",
                "message": "done",
                "llm_call_count": 1,
            }
        ],
    }


class TestResolveTask:
    @staticmethod
    def _native_warnings(caplog: pytest.LogCaptureFixture) -> list[str]:
        return [
            record.getMessage()
            for record in caplog.records
            if record.name == bmod.__name__ and record.levelno >= logging.WARNING
        ]

    def test_local_path_does_not_warn(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING, logger=bmod.__name__):
            cfg = resolve_task(_request({"harbor_task": "/tmp/some/task"}))
        assert cfg.path == Path("/tmp/some/task")
        assert cfg.name is None
        assert self._native_warnings(caplog) == []

    @pytest.mark.parametrize(
        "ref",
        ["3", "sha256:0123456789abcdef"],
    )
    def test_package_with_pinned_ref_does_not_warn(
        self, ref: str, caplog: pytest.LogCaptureFixture
    ):
        with caplog.at_level(logging.WARNING, logger=bmod.__name__):
            cfg = resolve_task(_request({"harbor_task": f"harbor/hello-world@{ref}"}))
        assert cfg.name == "harbor/hello-world"
        assert cfg.ref == ref
        assert cfg.path is None
        assert self._native_warnings(caplog) == []

    @pytest.mark.parametrize(
        "task_ref",
        ["harbor/hello-world", "harbor/hello-world@latest"],
    )
    def test_unpinned_package_warns_and_still_resolves_latest(
        self, task_ref: str, caplog: pytest.LogCaptureFixture
    ):
        with caplog.at_level(logging.WARNING, logger=bmod.__name__):
            cfg = resolve_task(_request({"harbor_task": task_ref}))
        assert cfg.ref == "latest"
        warnings = self._native_warnings(caplog)
        assert len(warnings) == 1
        assert "mutable ref 'latest'" in warnings[0]
        assert "sha256 digest" in warnings[0]

    def test_git_form_with_commit_does_not_warn(self, caplog: pytest.LogCaptureFixture):
        with caplog.at_level(logging.WARNING, logger=bmod.__name__):
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
        assert self._native_warnings(caplog) == []

    @pytest.mark.parametrize("commit", [None, "", "  \t"])
    def test_unpinned_git_warns_without_logging_url(
        self, commit: str | None, caplog: pytest.LogCaptureFixture
    ):
        git_url = "https://secret@example.com/private.git"
        metadata = {
            "harbor_task": "git",
            "git_url": git_url,
            "task_path": "tasks/foo",
        }
        if commit is not None:
            metadata["git_commit_id"] = commit
        with caplog.at_level(logging.WARNING, logger=bmod.__name__):
            cfg = resolve_task(_request(metadata))

        assert cfg.git_commit_id == commit
        warnings = self._native_warnings(caplog)
        assert len(warnings) == 1
        assert "unpinned git task" in warnings[0]
        assert "git_commit_id" in warnings[0]
        assert git_url not in warnings[0]

    def test_missing_raises(self):
        with pytest.raises(ValueError, match="harbor_task"):
            resolve_task(_request({}))

    def test_bare_package_name_without_org_raises(self):
        with pytest.raises(ValueError, match="org/name"):
            resolve_task(_request({"harbor_task": "helloworld"}))


class TestAgentConfig:
    def test_in_process_url_and_endpoint_wiring(self):
        backend = NativeHarborBackend()
        ac = backend._build_agent_config(_request(), _ctx())
        assert ac.name == "terminus-2"
        assert ac.kwargs["api_base"] == "http://ctrl:8080"
        assert "collect_rollout_details" not in ac.kwargs
        assert ac.kwargs["llm_kwargs"]["api_key"] == "sk-test"
        assert "extra_headers" not in ac.kwargs["llm_kwargs"]
        assert ac.kwargs["llm_kwargs"]["extra_body"] == {"stream": False}

    def test_unknown_builtin_fails_without_validated_binding(self):
        with pytest.raises(ValueError, match="no validated Native Harbor binding"):
            NativeHarborBackend(agent_name="nop")

    def test_agent_kwargs_override_terminus_default(self):
        backend = NativeHarborBackend(
            agent_kwargs={"proactive_summarization_threshold": 4000}
        )
        ac = backend._build_agent_config(_request(), _ctx())
        assert ac.kwargs["proactive_summarization_threshold"] == 4000
        assert ac.kwargs["enable_summarize"] is False

    def test_agent_kwargs_cannot_override_sdk_wiring(self):
        backend = NativeHarborBackend(agent_kwargs={"api_base": "http://evil"})
        ac = backend._build_agent_config(_request(), _ctx())
        assert ac.kwargs["api_base"] == "http://ctrl:8080"

    def test_agent_kwargs_llm_kwargs_deep_merged(self):
        backend = NativeHarborBackend(
            agent_kwargs={"llm_kwargs": {"timeout": 30, "extra_body": {"foo": 1}}}
        )
        ac = backend._build_agent_config(_request(), _ctx())
        llm = ac.kwargs["llm_kwargs"]
        assert llm["timeout"] == 30
        assert llm["api_key"] == "sk-test"
        assert llm["extra_body"] == {"foo": 1, "stream": False}

    def test_agent_kwargs_stream_is_binding_owned(self):
        backend = NativeHarborBackend(
            agent_kwargs={"llm_kwargs": {"extra_body": {"stream": True}}}
        )
        ac = backend._build_agent_config(_request(), _ctx())
        assert ac.kwargs["llm_kwargs"]["extra_body"]["stream"] is False

    def test_agent_env_passthrough(self):
        backend = NativeHarborBackend(agent_env={"FOO": "bar"})
        ac = backend._build_agent_config(_request(), _ctx())
        assert ac.env == {"FOO": "bar"}

    def test_agent_name_and_import_path_mutually_exclusive(self):
        with pytest.raises(ValueError, match="not both"):
            NativeHarborBackend(
                agent_name="terminus-2", agent_import_path="my.pkg:MyAgent"
            )

    @pytest.mark.parametrize(
        "agent_name",
        ["codex", "opencode", "claude-code"],
    )
    def test_agents_the_trainer_cannot_use_are_not_registered(self, agent_name: str):
        with pytest.raises(ValueError, match="no validated Native Harbor binding"):
            NativeHarborBackend(agent_name=agent_name)

    def test_registered_bindings_are_exactly_the_training_parity_set(self):
        assert sorted(_AGENT_BINDINGS) == [
            "custom-chat-completions",
            "custom-installed-chat-completions",
            "oracle",
            "terminus-2",
        ]

    def test_every_registered_binding_speaks_a_reachable_protocol(self):
        for binding in _AGENT_BINDINGS.values():
            assert binding.protocol in {
                _AgentProtocol.CHAT_COMPLETIONS,
                _AgentProtocol.NONE,
            }

    def test_every_model_driving_binding_is_training_supported(self):
        for binding in _AGENT_BINDINGS.values():
            if binding.emits_model_traffic:
                assert binding.training_supported, binding.name

    def test_custom_agent_is_wired_but_not_injected(self):
        with pytest.warns(UserWarning, match="custom Chat Completions"):
            backend = NativeHarborBackend(
                agent_import_path="my.custom.pkg:CustomAgent",
                binding="custom-chat-completions",
            )
        ac = backend._build_agent_config(_request(), _ctx())
        assert ac.import_path == "my.custom.pkg:CustomAgent"
        assert ac.name is None
        assert ac.kwargs["api_base"] == "http://ctrl:8080"
        assert ac.kwargs["llm_kwargs"]["api_key"] == "sk-test"
        assert "enable_summarize" not in ac.kwargs

    def test_custom_installed_agent_is_wired_through_env(self):
        with pytest.warns(UserWarning, match="custom installed Chat Completions"):
            backend = NativeHarborBackend(
                agent_import_path="my.custom.pkg:CustomInstalledAgent",
                binding="custom-installed-chat-completions",
            )
        ac = backend._build_agent_config(_request(), _ctx())
        assert ac.import_path == "my.custom.pkg:CustomInstalledAgent"
        assert ac.env["OPENAI_BASE_URL"] == "http://ctrl:8080"
        assert ac.env["OPENAI_API_KEY"] == "sk-test"
        assert ac.kwargs == {}

    def test_custom_installed_agent_passes_other_provider_credentials_through(self):
        with pytest.warns(UserWarning, match="custom installed Chat Completions"):
            backend = NativeHarborBackend(
                agent_import_path="my.custom.pkg:CustomInstalledAgent",
                binding="custom-installed-chat-completions",
                agent_env={
                    "ANTHROPIC_API_KEY": "user-anthropic",
                    "GEMINI_API_KEY": "user-gemini",
                    "EMAIL_SUBAGENT_MODEL": "anthropic:claude",
                },
            )
        ac = backend._build_agent_config(_request(), _ctx())
        assert ac.env["ANTHROPIC_API_KEY"] == "user-anthropic"
        assert ac.env["GEMINI_API_KEY"] == "user-gemini"
        assert ac.env["EMAIL_SUBAGENT_MODEL"] == "anthropic:claude"
        assert ac.env["OPENAI_BASE_URL"] == "http://ctrl:8080"

    @pytest.mark.parametrize("identity_key", ["OPENAI_BASE_URL", "OPENAI_API_KEY"])
    def test_custom_installed_agent_rejects_owned_identity_env(self, identity_key: str):
        with pytest.raises(ValueError, match=r"owns agent\.env identity keys"):
            NativeHarborBackend(
                agent_import_path="my.custom.pkg:CustomInstalledAgent",
                binding="custom-installed-chat-completions",
                agent_env={identity_key: "user-owned"},
            )

    def test_custom_agent_requires_an_explicit_custom_binding(self):
        with pytest.raises(ValueError, match="requires an explicit custom binding"):
            NativeHarborBackend(agent_import_path="my.custom.pkg:CustomAgent")

    def test_custom_agent_rejects_a_builtin_binding_name(self):
        with pytest.raises(ValueError, match="only supports the custom bindings"):
            NativeHarborBackend(
                agent_import_path="my.custom.pkg:CustomAgent",
                binding="terminus-2",
            )

    @pytest.mark.parametrize(
        ("import_path", "agent_name"),
        [
            ("harbor.agents.installed.opencode:OpenCode", "opencode"),
            ("harbor.agents.installed.codex:Codex", "codex"),
            ("harbor.agents.installed.claude_code:ClaudeCode", "claude-code"),
        ],
    )
    def test_import_path_cannot_reintroduce_an_unregistered_builtin(
        self, import_path: str, agent_name: str
    ):
        with pytest.raises(ValueError, match=rf"built-in '{agent_name}'.*no Native"):
            NativeHarborBackend(
                agent_import_path=import_path,
                binding="custom-chat-completions",
            )

    def test_import_path_cannot_bypass_a_registered_builtin_binding(self):
        with pytest.raises(ValueError, match=r"built-in 'oracle'.*agent_name"):
            NativeHarborBackend(
                agent_import_path="harbor.agents.oracle:OracleAgent",
                binding="custom-chat-completions",
            )

    def test_oracle_is_admitted_as_a_non_model_binding(self):
        with pytest.warns(UserWarning, match="oracle.*not training-safe"):
            backend = NativeHarborBackend(agent_name="oracle")

        ac = backend._build_agent_config(
            _request(), RolloutContext(chat_completions_url="", api_key=None)
        )

        assert ac.name == "oracle"
        assert ac.kwargs == {}
        assert ac.env == {}
        assert backend.health()["training_supported"] is False

    def test_resolved_agent_identity_is_read_only(self):
        backend = NativeHarborBackend()

        with pytest.raises(AttributeError):
            backend.agent_name = "claude-code"  # type: ignore[misc]
        with pytest.raises(AttributeError):
            backend.binding = "claude-code"  # type: ignore[misc]

    def test_chat_binding_rejects_non_openai_model_override(self):
        backend = NativeHarborBackend()
        request = _request(
            {"harbor_task": "/tmp/task", "harbor_model": "anthropic/claude"}
        )

        with pytest.raises(ValueError, match=r"requires a model prefixed.*openai"):
            backend._build_agent_config(request, _ctx())

    def test_missing_endpoint_raises(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.delenv("OSMOSIS_CHAT_COMPLETIONS_URL", raising=False)
        ctx = RolloutContext(chat_completions_url="", api_key="sk-test")
        backend = NativeHarborBackend()
        with pytest.raises(ValueError, match="no chat_completions_url"):
            backend._build_agent_config(_request(), ctx)

    def test_metadata_overrides_model(self):
        backend = NativeHarborBackend()
        md = {"harbor_task": "/tmp/task", "harbor_model": "openai/custom"}
        ac = backend._build_agent_config(_request(md), _ctx())
        assert ac.name == "terminus-2"
        assert ac.model_name == "openai/custom"

    @pytest.mark.parametrize("invalid_model", [None, "", "   ", 0, False])
    def test_explicit_invalid_metadata_model_is_rejected(self, invalid_model: Any):
        backend = NativeHarborBackend()
        request = _request({"harbor_task": "/tmp/task", "harbor_model": invalid_model})

        with pytest.raises(ValueError, match=r"harbor_model.*non-empty string"):
            backend._build_agent_config(request, _ctx())

    def test_agent_timeout_forwarded(self):
        backend = NativeHarborBackend()
        ac = backend._build_agent_config(_request(agent_timeout_sec=42.0), _ctx())
        assert ac.override_timeout_sec == 42.0

    def test_agent_setup_timeout_stored_and_forwarded(self):
        backend = NativeHarborBackend(agent_setup_timeout_sec=180.5)

        ac = backend._build_agent_config(_request(), _ctx())

        assert backend.agent_setup_timeout_sec == 180.5
        assert ac.override_setup_timeout_sec == 180.5

    def test_agent_setup_timeout_is_separate_from_run_timeout(self):
        backend = NativeHarborBackend(agent_setup_timeout_sec=180.0)

        ac = backend._build_agent_config(
            _request(agent_timeout_sec=42.0),
            _ctx(),
        )

        assert ac.override_setup_timeout_sec == 180.0
        assert ac.override_timeout_sec == 42.0

    @pytest.mark.parametrize(
        "invalid_timeout",
        [0.0, -0.1, float("inf"), float("nan")],
    )
    def test_agent_setup_timeout_must_be_positive_and_finite(
        self, invalid_timeout: float
    ):
        with pytest.raises(ValueError, match="agent_setup_timeout_sec must be > 0"):
            NativeHarborBackend(agent_setup_timeout_sec=invalid_timeout)


class TestFullConfigConstructor:
    def test_full_configs_are_preserved_and_deep_cloned_per_rollout(self):
        agent = AgentConfig(
            name="terminus-2",
            model_name="openai/config-model",
            skills=["org/skill@1"],
            mcp_servers=[
                MCPServerConfig(
                    name="shell-tools",
                    transport="stdio",
                    command="serve-tools",
                    args=["--safe"],
                )
            ],
            include_logs=["*.json"],
            exclude_logs=["debug-*"],
            extra_allowed_hosts=["models.example.com"],
            override_setup_timeout_sec=11.0,
            override_timeout_sec=12.0,
            max_timeout_sec=13.0,
            kwargs={"nested": {"items": ["original"]}},
            env={"CUSTOM_TOKEN": "agent-secret"},
        )
        environment = EnvironmentConfig(
            type=EnvironmentType.DAYTONA,
            force_build=True,
            delete=False,
            override_cpus=4,
            env={"SANDBOX_TOKEN": "environment-secret"},
            kwargs={"nested": {"region": "us-west"}},
            extra_allowed_hosts=["packages.example.com"],
        )
        verifier = VerifierConfig(
            override_timeout_sec=21.0,
            max_timeout_sec=22.0,
            include_logs=["reward.json"],
            exclude_logs=["verbose.log"],
            env={"VERIFIER_TOKEN": "verifier-secret"},
            import_path="my.verifier:Verifier",
            kwargs={"nested": {"threshold": 0.75}},
            disable=True,
        )
        backend = NativeHarborBackend(
            agent=agent,
            environment=environment,
            verifier=verifier,
        )

        agent_one = backend._build_agent_config(_request(), _ctx())
        agent_two = backend._build_agent_config(_request(), _ctx())
        trial_one = backend._build_trial_config(
            _request(), TaskConfig(path=Path("/tmp/task")), agent_one, "trial-one"
        )
        trial_two = backend._build_trial_config(
            _request(), TaskConfig(path=Path("/tmp/task")), agent_two, "trial-two"
        )

        assert agent_one.model_name == "openai/config-model"
        assert agent_one.skills == ["org/skill@1"]
        assert agent_one.mcp_servers[0].command == "serve-tools"
        assert agent_one.include_logs == ["*.json"]
        assert agent_one.exclude_logs == ["debug-*"]
        assert agent_one.extra_allowed_hosts == ["models.example.com"]
        assert agent_one.override_setup_timeout_sec == 11.0
        assert agent_one.override_timeout_sec == 12.0
        assert agent_one.max_timeout_sec == 13.0
        assert agent_one.env["CUSTOM_TOKEN"] == "agent-secret"
        assert trial_one.environment.type == EnvironmentType.DAYTONA
        assert trial_one.environment.force_build is True
        assert trial_one.environment.delete is False
        assert trial_one.environment.override_cpus == 4
        assert trial_one.environment.env["SANDBOX_TOKEN"] == "environment-secret"
        assert trial_one.verifier.disable is False
        assert trial_one.verifier.override_timeout_sec == 21.0
        assert trial_one.verifier.max_timeout_sec == 22.0
        assert trial_one.verifier.include_logs == ["reward.json"]
        assert trial_one.verifier.exclude_logs == ["verbose.log"]
        assert trial_one.verifier.env["VERIFIER_TOKEN"] == "verifier-secret"
        assert trial_one.verifier.import_path == "my.verifier:Verifier"

        agent_one.kwargs["nested"]["items"].append("mutated")
        agent_one.skills.append("mutated-skill")
        agent_one.env["CUSTOM_TOKEN"] = "mutated"
        trial_one.environment.kwargs["nested"]["region"] = "mutated"
        trial_one.verifier.kwargs["nested"]["threshold"] = 0.0

        assert agent.kwargs == {"nested": {"items": ["original"]}}
        assert agent.skills == ["org/skill@1"]
        assert agent.env["CUSTOM_TOKEN"] == "agent-secret"
        assert environment.kwargs == {"nested": {"region": "us-west"}}
        assert environment.env["SANDBOX_TOKEN"] == "environment-secret"
        assert verifier.kwargs == {"nested": {"threshold": 0.75}}
        assert verifier.disable is True
        assert agent_two.kwargs["nested"]["items"] == ["original"]
        assert agent_two.skills == ["org/skill@1"]
        assert agent_two.env["CUSTOM_TOKEN"] == "agent-secret"
        assert trial_two.environment.kwargs == {"nested": {"region": "us-west"}}
        assert trial_two.verifier.kwargs == {"nested": {"threshold": 0.75}}
        assert agent_one is not agent_two
        assert trial_one.environment is not trial_two.environment
        assert trial_one.verifier is not trial_two.verifier

    def test_model_and_timeout_ownership_overlays_preserve_safety_caps(self):
        agent = AgentConfig(
            name="terminus-2",
            model_name="openai/agent-default",
            override_setup_timeout_sec=10.0,
            override_timeout_sec=20.0,
            max_timeout_sec=30.0,
        )
        verifier = VerifierConfig(
            override_timeout_sec=40.0,
            max_timeout_sec=50.0,
        )
        backend = NativeHarborBackend(
            agent=agent,
            verifier=verifier,
            model_name="openai/constructor-default",
            agent_setup_timeout_sec=15.0,
        )
        request = _request(
            {
                "harbor_task": "/tmp/task",
                "harbor_model": "openai/row-model",
            },
            agent_timeout_sec=25.0,
            grader_timeout_sec=45.0,
        )

        rollout_agent = backend._build_agent_config(request, _ctx())
        rollout = backend._build_trial_config(
            request,
            TaskConfig(path=Path("/tmp/task")),
            rollout_agent,
            "trial-overlays",
        )

        assert rollout.agent.model_name == "openai/row-model"
        assert rollout.agent.override_setup_timeout_sec == 15.0
        assert rollout.agent.override_timeout_sec == 25.0
        assert rollout.agent.max_timeout_sec == 30.0
        assert rollout.verifier.override_timeout_sec == 45.0
        assert rollout.verifier.max_timeout_sec == 50.0
        assert agent.model_name == "openai/agent-default"
        assert agent.override_setup_timeout_sec == 10.0
        assert agent.override_timeout_sec == 20.0
        assert verifier.override_timeout_sec == 40.0

        default_request = _request()
        default_agent = backend._build_agent_config(default_request, _ctx())
        default_trial = backend._build_trial_config(
            default_request,
            TaskConfig(path=Path("/tmp/task")),
            default_agent,
            "trial-defaults",
        )
        assert default_agent.model_name == "openai/constructor-default"
        assert default_agent.override_timeout_sec == 20.0
        assert default_trial.verifier.override_timeout_sec == 40.0

    @pytest.mark.parametrize(
        ("agent", "message"),
        [
            (
                AgentConfig(name="terminus-2", n_concurrent=1),
                "agent.n_concurrent is unsupported",
            ),
            (
                AgentConfig(name="terminus-2", concurrency_group="shared"),
                "agent.concurrency_group is unsupported",
            ),
            (
                AgentConfig(name="terminus-2", resume_trajectory=True),
                "agent.resume_trajectory is unsupported",
            ),
        ],
    )
    def test_agent_owned_fields_are_rejected(
        self, agent: AgentConfig, message: str
    ) -> None:
        with pytest.raises(ValueError, match=message):
            NativeHarborBackend(agent=agent)

    def test_canonical_and_legacy_config_inputs_cannot_be_mixed(self):
        with pytest.raises(ValueError, match="agent cannot be combined"):
            NativeHarborBackend(
                agent=AgentConfig(name="terminus-2"),
                agent_kwargs={"temperature": 0},
            )
        with pytest.raises(ValueError, match="cannot both be set"):
            NativeHarborBackend(
                environment=EnvironmentConfig(type=EnvironmentType.DOCKER),
                environment_config=EnvironmentConfig(type=EnvironmentType.DAYTONA),
            )

    def test_managed_skypilot_placement_does_not_mutate_caller(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HARBOR_SKYPILOT_CONTEXT", "managed-cluster")
        environment = EnvironmentConfig(
            type=EnvironmentType.SKYPILOT,
            kwargs={"nested": {"value": "preserved"}},
        )
        backend = NativeHarborBackend(environment=environment)
        agent = backend._build_agent_config(_request(), _ctx())
        trial = backend._build_trial_config(
            _request(), TaskConfig(path=Path("/tmp/task")), agent, "trial-sky"
        )

        assert "context_name" not in environment.kwargs
        assert trial.environment.kwargs["context_name"] == "managed-cluster"
        assert trial.environment.kwargs["nested"] == {"value": "preserved"}

    def test_explicit_empty_agent_config_preserves_oracle_default(self):
        agent = AgentConfig()
        with pytest.warns(UserWarning, match="oracle.*not training-safe"):
            backend = NativeHarborBackend(agent=agent)

        built = backend._build_agent_config(
            _request(), RolloutContext(chat_completions_url="", api_key=None)
        )

        assert agent.name == "oracle"
        assert built.name == "oracle"
        assert backend.agent_name == "oracle"


class TestPrewarm:
    async def test_builds_install_only_configs_without_rollout_context(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        task = TaskConfig(
            path=Path("/tmp/prewarm-task"),
            overwrite=True,
            source="prewarm-test",
        )
        agent = AgentConfig(
            name="terminus-2",
            model_name="openai/prewarm-model",
            skills=["org/skill@sha256:" + "a" * 64],
            kwargs={"nested": {"items": ["original"]}},
            env={"CUSTOM_TOKEN": "agent-secret"},
            override_setup_timeout_sec=31.0,
        )
        environment = EnvironmentConfig(
            type=EnvironmentType.DAYTONA,
            delete=False,
            kwargs={"nested": {"region": "us-west"}},
        )
        verifier = VerifierConfig(
            env={"VERIFIER_TOKEN": "verifier-secret"},
            kwargs={"nested": {"threshold": 0.75}},
        )
        captured: list[Any] = []
        queues: list[Any] = []

        async def _submit(queue: Any, trial_config: Any) -> Any:
            queues.append(queue)
            captured.append(trial_config)
            return _trial_result()

        monkeypatch.setattr(bmod.TrialQueue, "submit", _submit)
        backend = NativeHarborBackend(
            agent=agent,
            environment=environment,
            verifier=verifier,
            trials_dir=tmp_path,
            cleanup_successful_trials=False,
        )

        await backend.prewarm([task, task])

        assert queues == [backend._queue, backend._queue]
        assert backend._queue._retry_config.max_retries == 0
        assert len(captured) == 2
        assert len({config.trial_name for config in captured}) == 2
        for config in captured:
            assert config.trial_name.startswith("native-prewarm-")
            assert config.install_only is True
            assert config.verifier.disable is True
            assert config.task is not task
            assert config.task.source == "prewarm-test"
            assert config.agent is not agent
            assert config.environment is not environment
            assert config.verifier is not verifier
            assert config.agent.model_name == "openai/prewarm-model"
            assert config.agent.skills == ["org/skill@sha256:" + "a" * 64]
            assert config.agent.env == {"CUSTOM_TOKEN": "agent-secret"}
            assert config.agent.override_setup_timeout_sec == 31.0
            assert "api_base" not in config.agent.kwargs
            assert "llm_kwargs" not in config.agent.kwargs
            assert config.environment.type == EnvironmentType.DAYTONA
            assert config.environment.delete is False
            assert config.verifier.env == {"VERIFIER_TOKEN": "verifier-secret"}

        captured[0].task.path = Path("/tmp/mutated")
        captured[0].agent.kwargs["nested"]["items"].append("mutated")
        captured[0].environment.kwargs["nested"]["region"] = "mutated"
        captured[0].verifier.kwargs["nested"]["threshold"] = 0.0

        assert task.path == Path("/tmp/prewarm-task")
        assert agent.kwargs == {"nested": {"items": ["original"]}}
        assert environment.kwargs == {"nested": {"region": "us-west"}}
        assert verifier.kwargs == {"nested": {"threshold": 0.75}}
        assert verifier.disable is False
        assert backend._verifier_config.disable is False
        assert captured[1].task.path == Path("/tmp/prewarm-task")
        assert captured[1].agent.kwargs["nested"] == {"items": ["original"]}
        assert captured[1].environment.kwargs == {"nested": {"region": "us-west"}}
        assert captured[1].verifier.kwargs == {"nested": {"threshold": 0.75}}

    async def test_prewarm_installs_agent_without_per_rollout_identity(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        async def _submit(_queue: Any, trial_config: Any) -> Any:
            captured["config"] = trial_config
            return _trial_result()

        monkeypatch.setattr(bmod.TrialQueue, "submit", _submit)
        backend = NativeHarborBackend(cleanup_successful_trials=False)

        await backend.prewarm([TaskConfig(path=Path("/tmp/task"))])

        config = captured["config"]
        assert config.agent.kwargs["enable_summarize"] is False
        assert "api_base" not in config.agent.kwargs
        assert "llm_kwargs" not in config.agent.kwargs
        assert config.agent.env == {}

    async def test_attempts_all_tasks_aggregates_failures_and_cleans_only_successes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        tasks = [
            TaskConfig(path=Path("/tmp/success")),
            TaskConfig(path=Path("/tmp/raised")),
            TaskConfig(path=Path("/tmp/reported")),
        ]
        attempted: list[str] = []
        configs: dict[str, Any] = {}

        async def _submit(_queue: Any, trial_config: Any) -> Any:
            task_name = trial_config.task.path.name
            attempted.append(task_name)
            configs[task_name] = trial_config
            (tmp_path / trial_config.trial_name).mkdir()
            if task_name == "raised":
                raise OSError("image builder unavailable")
            if task_name == "reported":
                return _trial_result(
                    exc_message="agent install failed",
                    exc_type="AgentSetupError",
                )
            return _trial_result()

        monkeypatch.setattr(bmod.TrialQueue, "submit", _submit)
        backend = NativeHarborBackend(trials_dir=tmp_path)

        with pytest.raises(RuntimeError) as exc_info:
            await backend.prewarm(tasks)

        assert sorted(attempted) == ["raised", "reported", "success"]
        message = str(exc_info.value)
        assert "failed for 2 of 3 task(s)" in message
        assert "/tmp/raised [OSError]; inspect preserved trial" in message
        assert "/tmp/reported [AgentSetupError]; inspect preserved trial" in message
        assert "image builder unavailable" not in message
        assert "agent install failed" not in message
        assert not (tmp_path / configs["success"].trial_name).exists()
        assert (tmp_path / configs["raised"].trial_name).is_dir()
        assert (tmp_path / configs["reported"].trial_name).is_dir()

    async def test_rejects_empty_task_list(self) -> None:
        backend = NativeHarborBackend()

        with pytest.raises(ValueError, match="at least one Harbor TaskConfig"):
            await backend.prewarm([])
        with pytest.raises(ValueError, match="at least one Harbor TaskConfig"):
            backend.prewarm_lifespan([])

    async def test_failure_aggregate_omits_raw_setup_output_and_credentials(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        raised_url = "https://raised:secret@example.com/private.git"
        reported_url = "https://reported:secret@example.com/private.git"
        tasks = [
            TaskConfig(
                git_url=raised_url,
                path=Path("tasks/raised"),
                git_commit_id="abc123",
            ),
            TaskConfig(
                git_url=reported_url,
                path=Path("tasks/reported"),
                git_commit_id="def456",
            ),
        ]

        agent_env_secret = "agent-env-secret"
        agent_kwarg_secret = "agent-kwarg-secret"

        async def _submit(_queue: Any, trial_config: Any) -> Any:
            if trial_config.task.path.name == "raised":
                raise OSError(f"clone of {raised_url} failed with {agent_env_secret}")
            return _trial_result(
                exc_message=(
                    f"download of {reported_url} failed with {agent_kwarg_secret}"
                ),
                exc_type="TaskDownloadError",
            )

        monkeypatch.setattr(bmod.TrialQueue, "submit", _submit)
        backend = NativeHarborBackend(
            agent_env={"CUSTOM_TOKEN": agent_env_secret},
            agent_kwargs={"llm_kwargs": {"api_key": agent_kwarg_secret}},
        )

        with caplog.at_level(logging.ERROR):
            with pytest.raises(RuntimeError) as exc_info:
                await backend.prewarm(tasks)

        message = str(exc_info.value)
        assert "git:tasks/raised@abc123 [OSError]" in message
        assert "git:tasks/reported@def456 [TaskDownloadError]" in message
        assert message.count("no trial directory was created") == 2
        for secret in (
            "raised:secret",
            "reported:secret",
            agent_env_secret,
            agent_kwarg_secret,
        ):
            assert secret not in message
            assert secret not in caplog.text

    async def test_lifespan_clones_tasks_and_awaits_prewarm_before_serving(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from osmosis_ai.rollout.server import create_rollout_server

        backend = NativeHarborBackend()
        task = TaskConfig(path=Path("/tmp/original"))
        events: list[str] = []
        received: list[TaskConfig] = []

        async def _prewarm(tasks: Any) -> None:
            events.append("prewarm")
            received.extend(tasks)

        monkeypatch.setattr(backend, "prewarm", _prewarm)
        app = create_rollout_server(
            backend=backend,
            lifespan=backend.prewarm_lifespan([task]),
        )
        task.path = Path("/tmp/mutated-after-app-creation")

        assert events == []
        async with app.router.lifespan_context(app):
            assert events == ["prewarm"]
            assert received[0].path == Path("/tmp/original")
            events.append("serving")
        assert events == ["prewarm", "serving"]

    async def test_lifespan_failure_aborts_startup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from osmosis_ai.rollout.server import create_rollout_server

        backend = NativeHarborBackend()

        async def _prewarm(_tasks: Any) -> None:
            raise RuntimeError("prewarm failed")

        monkeypatch.setattr(backend, "prewarm", _prewarm)
        app = create_rollout_server(
            backend=backend,
            lifespan=backend.prewarm_lifespan([TaskConfig(path=Path("/tmp/task"))]),
        )
        entered = False

        with pytest.raises(RuntimeError, match="prewarm failed"):
            async with app.router.lifespan_context(app):
                entered = True
        assert entered is False


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

        assert gr_result.sample.reward == 1.0
        assert "extra_headers" not in capture["config"].agent.kwargs["llm_kwargs"]
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
        assert on_wf.call_args.args[0].extra_fields["phase"] == "setup"
        assert (
            on_wf.call_args.args[0].extra_fields["harbor_exception_type"]
            == "RuntimeError"
        )
        assert on_gr.call_args.args[0].status == RolloutStatus.FAILURE

    @pytest.mark.parametrize("queue_fails", [False, True])
    async def test_controller_identity_is_wired_per_rollout(
        self,
        monkeypatch: pytest.MonkeyPatch,
        queue_fails: bool,
    ) -> None:
        backend = NativeHarborBackend()
        captured: dict[str, Any] = {}

        async def _submit(queue: Any, trial_config: Any) -> Any:
            captured["agent"] = trial_config.agent
            if queue_fails:
                raise RuntimeError("trial failed")
            return _trial_result(rewards={"reward": 1.0})

        monkeypatch.setattr(bmod.TrialQueue, "submit", _submit)
        with _ctx():
            await backend.execute(_request(), AsyncMock(), AsyncMock())

        agent = captured["agent"]
        assert agent.kwargs["api_base"] == "http://ctrl:8080"
        assert agent.kwargs["llm_kwargs"]["api_key"] == "sk-test"

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
        assert on_wf.call_args.args[0].status == RolloutStatus.SUCCESS
        gr_result = on_gr.call_args.args[0]
        assert gr_result.status == RolloutStatus.FAILURE
        assert gr_result.err_category == RolloutErrorCategory.VALIDATION_ERROR

    async def test_reward_does_not_revive_failed_trial(self, monkeypatch):
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
        workflow_result = on_wf.call_args.args[0]
        gr_result = on_gr.call_args.args[0]
        assert workflow_result.status == RolloutStatus.FAILURE
        assert workflow_result.sample.reward is None
        assert gr_result.status == RolloutStatus.FAILURE
        assert gr_result.sample.reward is None

    async def test_swallowed_exception_is_agent_error(self, monkeypatch):
        # Harbor reports in-trial failures through exception_info.
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
        assert on_wf.call_args.args[0].err_category == RolloutErrorCategory.AGENT_ERROR

    async def test_hook_phase_timings_and_error_payload_are_reused(
        self, monkeypatch, caplog
    ):
        ticks = iter(float(value) for value in range(20))
        monkeypatch.setattr(bmod, "monotonic", lambda: next(ticks))
        started_at = datetime(2026, 1, 1, tzinfo=UTC)
        result = _trial_result(
            rewards={"reward": 0.7},
            exc_message="agent timed out",
            exc_type="AgentTimeoutError",
        )
        result.started_at = started_at
        result.finished_at = started_at + timedelta(seconds=10)
        result.environment_setup = SimpleNamespace(
            started_at=started_at,
            finished_at=started_at + timedelta(seconds=2),
        )
        result.agent_setup = SimpleNamespace(
            started_at=started_at + timedelta(seconds=2),
            finished_at=started_at + timedelta(seconds=3),
        )
        result.agent_execution = SimpleNamespace(
            started_at=started_at + timedelta(seconds=3),
            finished_at=started_at + timedelta(seconds=6),
        )
        result.verifier = SimpleNamespace(
            started_at=started_at + timedelta(seconds=6),
            finished_at=started_at + timedelta(seconds=10),
        )

        async def _submit(queue: Any, trial_config: Any) -> Any:
            for event_name in (
                "start",
                "environment-start",
                "agent-start",
                "agent-end",
                "verification-start",
                "end",
            ):
                event = next(
                    event for event in queue._hooks if event.value == event_name
                )
                hook_event = SimpleNamespace(
                    trial_name=trial_config.trial_name,
                    result=result,
                )
                for callback in queue._hooks[event]:
                    await callback(hook_event)
            return result

        monkeypatch.setattr(bmod.TrialQueue, "submit", _submit)
        caplog.set_level(
            logging.ERROR,
            logger="osmosis_ai.rollout.backend.native_harbor.backend",
        )
        on_wf, on_gr = AsyncMock(), AsyncMock()

        with _ctx():
            await NativeHarborBackend().execute(_request(), on_wf, on_gr)

        workflow_payload = on_wf.call_args.args[0].extra_fields
        grader_payload = on_gr.call_args.args[0].extra_fields
        assert grader_payload == workflow_payload
        assert workflow_payload == {
            "backend": "native_harbor",
            "phase": "agent",
            "harbor_exception_type": "AgentTimeoutError",
            "category": "agent_error",
            "timings_sec": {
                "setup": 1.0,
                "trial_setup": 1.0,
                "environment_setup": 2.0,
                "agent_setup": 1.0,
                "agent": 3.0,
                "verification": 4.0,
                "trial": 10.0,
            },
        }
        assert json.dumps(workflow_payload, sort_keys=True) in caplog.text

    async def test_late_verifier_failure_keeps_callback_timing_and_phase(
        self, monkeypatch
    ):
        result = _trial_result(rewards={"reward": 0.7})

        async def _submit(queue: Any, trial_config: Any) -> Any:
            hook_event = SimpleNamespace(
                trial_name=trial_config.trial_name,
                result=result,
            )
            verification_event = next(
                event for event in queue._hooks if event.value == "verification-start"
            )
            for callback in queue._hooks[verification_event]:
                await callback(hook_event)

            result.exception_info = SimpleNamespace(
                exception_message="verifier timed out",
                exception_type="VerifierTimeoutError",
            )
            end_event = next(event for event in queue._hooks if event.value == "end")
            for callback in queue._hooks[end_event]:
                await callback(hook_event)
            return result

        monkeypatch.setattr(bmod.TrialQueue, "submit", _submit)
        on_wf, on_gr = AsyncMock(), AsyncMock()

        with _ctx():
            await NativeHarborBackend().execute(_request(), on_wf, on_gr)

        workflow_result = on_wf.call_args.args[0]
        grader_result = on_gr.call_args.args[0]
        assert workflow_result.status == RolloutStatus.SUCCESS
        assert workflow_result.extra_fields["harbor_exception_type"] is None
        assert grader_result.status == RolloutStatus.FAILURE
        assert grader_result.extra_fields["phase"] == "verification"
        assert (
            grader_result.extra_fields["harbor_exception_type"]
            == "VerifierTimeoutError"
        )

    async def test_late_failure_without_grader_callback_is_archived(
        self, monkeypatch, tmp_path
    ):
        result = _trial_result(rewards={"reward": 0.7})

        async def _submit(queue: Any, trial_config: Any) -> Any:
            hook_event = SimpleNamespace(
                trial_name=trial_config.trial_name,
                result=result,
            )
            verification_event = next(
                event for event in queue._hooks if event.value == "verification-start"
            )
            for callback in queue._hooks[verification_event]:
                await callback(hook_event)
            result.exception_info = SimpleNamespace(
                exception_message="verifier timed out",
                exception_type="VerifierTimeoutError",
            )
            end_event = next(event for event in queue._hooks if event.value == "end")
            for callback in queue._hooks[end_event]:
                await callback(hook_event)
            return result

        monkeypatch.setattr(bmod.TrialQueue, "submit", _submit)
        backend = NativeHarborBackend()
        backend.artifact_root = tmp_path
        on_wf = AsyncMock()

        with _ctx():
            await backend.execute(_request(), on_wf, None)

        assert on_wf.call_args.args[0].status == RolloutStatus.SUCCESS
        diagnostics = json.loads((tmp_path / "ROLL" / "diagnostics.json").read_text())
        assert diagnostics["phase"] == "verification"
        assert diagnostics["harbor_exception_type"] == "VerifierTimeoutError"
        assert diagnostics["category"] == "agent_error"

    async def test_post_callback_queue_exception_is_archived_and_graded(
        self, monkeypatch, tmp_path
    ):
        result = _trial_result(rewards={"reward": 0.7})

        async def _submit(queue: Any, trial_config: Any) -> Any:
            hook_event = SimpleNamespace(
                trial_name=trial_config.trial_name,
                result=result,
            )
            verification_event = next(
                event for event in queue._hooks if event.value == "verification-start"
            )
            for callback in queue._hooks[verification_event]:
                await callback(hook_event)
            raise RuntimeError("failed to finalize trial result")

        monkeypatch.setattr(bmod.TrialQueue, "submit", _submit)
        backend = NativeHarborBackend()
        backend.artifact_root = tmp_path
        on_wf, on_gr = AsyncMock(), AsyncMock()

        with _ctx():
            await backend.execute(_request(), on_wf, on_gr)

        assert on_wf.call_args.args[0].status == RolloutStatus.SUCCESS
        grader_result = on_gr.call_args.args[0]
        assert grader_result.status == RolloutStatus.FAILURE
        assert grader_result.extra_fields["phase"] == "verification"
        assert grader_result.extra_fields["harbor_exception_type"] == "RuntimeError"
        diagnostics = json.loads((tmp_path / "ROLL" / "diagnostics.json").read_text())
        assert diagnostics == grader_result.extra_fields

    async def test_grader_callback_failure_propagates_after_trial(self, monkeypatch):
        # The server owns the final notification fallback.
        _patch_trial(monkeypatch, result=_trial_result(rewards={"reward": 1.0}))
        backend = NativeHarborBackend()
        on_wf = AsyncMock()
        on_gr = AsyncMock(side_effect=RuntimeError("controller down"))
        with _ctx():
            with pytest.raises(RuntimeError, match="controller down"):
                await backend.execute(_request(), on_wf, on_gr)
        on_wf.assert_awaited_once()
        on_gr.assert_awaited_once()

    async def test_failed_verification_hook_callback_retries_after_trial(
        self, monkeypatch
    ):
        result = _trial_result(rewards={"reward": 1.0})

        async def _submit(queue: Any, trial_config: Any) -> Any:
            hook = next(
                callback
                for event, callbacks in queue._hooks.items()
                if event.value == "verification-start"
                for callback in callbacks
            )
            await hook(
                SimpleNamespace(trial_name=trial_config.trial_name, result=result)
            )
            return result

        monkeypatch.setattr(bmod.TrialQueue, "submit", _submit)
        on_wf = AsyncMock(side_effect=[RuntimeError("temporary outage"), None])

        with _ctx():
            await NativeHarborBackend().execute(_request(), on_wf)

        assert on_wf.await_count == 2

    async def test_single_step_workflow_callback_precedes_verifier_completion(
        self, monkeypatch
    ):
        events: list[str] = []
        result = _trial_result(rewards={"reward": 1.0})

        async def _submit(queue: Any, trial_config: Any) -> Any:
            hook = next(
                callback
                for event, callbacks in queue._hooks.items()
                if event.value == "verification-start"
                for callback in callbacks
            )
            await hook(
                SimpleNamespace(trial_name=trial_config.trial_name, result=result)
            )
            events.append("verifier-finished")
            return result

        async def _workflow_callback(result: Any) -> None:
            events.append("workflow")

        async def _grader_callback(result: Any) -> None:
            events.append("grader")

        monkeypatch.setattr(bmod.TrialQueue, "submit", _submit)
        with _ctx():
            await NativeHarborBackend().execute(
                _request(), _workflow_callback, _grader_callback
            )

        assert events == ["workflow", "verifier-finished", "grader"]


class TestConcurrencyAndLifecycle:
    def test_retry_config_is_not_a_constructor_argument(self):
        with pytest.raises(TypeError, match="retry_config"):
            NativeHarborBackend(retry_config=bmod.RetryConfig(max_retries=1))  # type: ignore[call-arg]

    def test_trial_queue_retries_are_hard_disabled(self):
        backend = NativeHarborBackend()
        assert backend._queue._retry_config.max_retries == 0

    def test_unbounded_concurrency_rejected(self):
        with pytest.raises(ValueError, match="max_concurrent must be >= 1"):
            NativeHarborBackend(max_concurrent=0)

    def test_negative_queue_depth_rejected(self):
        with pytest.raises(ValueError, match="max_queue_depth must be >= 0"):
            NativeHarborBackend(max_queue_depth=-1)

    def test_health_reports_capacity_and_binding_capabilities(self):
        backend = NativeHarborBackend(max_concurrent=3)
        assert backend.max_concurrency == 3
        assert backend.max_queue_depth == 3
        assert backend.health() == {
            "status": "ok",
            "backend": "native_harbor",
            "agent": "terminus-2",
            "binding": "terminus-2",
            "binding_protocol": "OpenAI Chat Completions",
            "protocol_capabilities": ["OpenAI Chat Completions"],
            "training_supported": True,
            "max_concurrency": 3,
            "max_queue_depth": 3,
        }

    async def test_successful_trial_dir_cleaned_up(self, monkeypatch, tmp_path):
        _patch_trial(monkeypatch, result=_trial_result(rewards={"reward": 1.0}))
        backend = NativeHarborBackend(trials_dir=tmp_path)
        trial_dir = tmp_path / "native-ROLL"
        trial_dir.mkdir()
        with _ctx():
            await backend.execute(_request(), AsyncMock(), AsyncMock())
        assert not trial_dir.exists()

    async def test_failed_trial_dir_kept(self, monkeypatch, tmp_path):
        _patch_trial(monkeypatch, result=_trial_result(exc_message="boom"))
        backend = NativeHarborBackend(trials_dir=tmp_path)
        trial_dir = tmp_path / "native-ROLL"
        trial_dir.mkdir()
        with _ctx():
            await backend.execute(_request(), AsyncMock(), AsyncMock())
        assert trial_dir.exists()

    async def test_native_atif_preserved_with_agent_extra_redacted(
        self, monkeypatch, tmp_path
    ):
        _patch_trial(monkeypatch, result=_trial_result(rewards={"reward": 1.0}))
        backend = NativeHarborBackend(trials_dir=tmp_path)
        backend.artifact_root = tmp_path / "saved"
        trajectory_path = tmp_path / "native-ROLL" / "agent" / "trajectory.json"
        trajectory_path.parent.mkdir(parents=True)
        trajectory_path.write_text(json.dumps(_native_trajectory()))
        on_wf, on_gr = AsyncMock(), AsyncMock()

        with _ctx():
            await backend.execute(_request(), on_wf, on_gr)

        workflow_document = on_wf.call_args.args[0].trajectory_document
        grader_document = on_gr.call_args.args[0].trajectory_document
        assert grader_document == workflow_document
        assert workflow_document["steps"] == _native_trajectory()["steps"]
        extra = workflow_document["agent"]["extra"]
        assert extra["llm_kwargs"] == {
            "api_key": "[REDACTED]",
            "temperature": 0,
        }
        assert extra["command"] == ["agent", "--token", "[REDACTED]"]
        assert extra["safe"] == "kept"
        saved_path = backend.artifact_root / "ROLL" / "trajectory.json"
        assert saved_path.is_file()
        saved = json.loads(saved_path.read_text())
        diagnostics = saved["extra"]["osmosis"]["result_extra_fields"]
        assert diagnostics["backend"] == "native_harbor"
        assert diagnostics["phase"] == "setup"
        assert diagnostics["harbor_exception_type"] is None
        assert diagnostics["category"] is None
        assert not trajectory_path.exists()

    async def test_native_atif_persistence_failure_preserves_successful_trial(
        self, monkeypatch, tmp_path
    ):
        _patch_trial(monkeypatch, result=_trial_result(rewards={"reward": 1.0}))
        backend = NativeHarborBackend(trials_dir=tmp_path)
        trajectory_path = tmp_path / "native-ROLL" / "agent" / "trajectory.json"
        trajectory_path.parent.mkdir(parents=True)
        trajectory_path.write_text(json.dumps(_native_trajectory()))
        persist = AsyncMock(return_value=False)
        monkeypatch.setattr(bmod, "_save_trajectories_with_status", persist)

        with _ctx():
            await backend.execute(_request(), AsyncMock(), AsyncMock())

        assert trajectory_path.exists()
        persist.assert_awaited_once()

    async def test_invalid_native_atif_preserves_successful_trial(
        self, monkeypatch, tmp_path
    ):
        _patch_trial(monkeypatch, result=_trial_result(rewards={"reward": 1.0}))
        backend = NativeHarborBackend(trials_dir=tmp_path)
        trajectory_path = tmp_path / "native-ROLL" / "agent" / "trajectory.json"
        trajectory_path.parent.mkdir(parents=True)
        trajectory_path.write_text('{"not": "atif"}')

        with _ctx():
            await backend.execute(_request(), AsyncMock(), AsyncMock())

        assert trajectory_path.exists()

    async def test_harbor_collected_artifacts_relocated_before_cleanup(
        self, monkeypatch, tmp_path
    ):
        _patch_trial(monkeypatch, result=_trial_result(rewards={"reward": 1.0}))
        trials_dir = tmp_path / "trials"
        artifact_root = tmp_path / "saved"
        backend = NativeHarborBackend(trials_dir=trials_dir)
        backend.artifact_root = artifact_root
        artifact = (
            trials_dir
            / "native-ROLL"
            / "artifacts"
            / "logs"
            / "artifacts"
            / "result.txt"
        )
        artifact.parent.mkdir(parents=True)
        artifact.write_text("user-selected output")

        with _ctx():
            await backend.execute(_request(), AsyncMock(), AsyncMock())

        relocated = (
            artifact_root / "ROLL" / "artifacts" / "logs" / "artifacts" / "result.txt"
        )
        assert relocated.read_text() == "user-selected output"
        assert not (trials_dir / "native-ROLL").exists()

    async def test_artifact_copy_failure_preserves_successful_trial(
        self, monkeypatch, tmp_path
    ):
        _patch_trial(monkeypatch, result=_trial_result(rewards={"reward": 1.0}))
        backend = NativeHarborBackend(trials_dir=tmp_path)
        source = tmp_path / "native-ROLL" / "artifacts" / "out.txt"
        source.parent.mkdir(parents=True)
        source.write_text("keep me")

        def _fail_copy(*args: Any, **kwargs: Any) -> int:
            raise OSError("destination unavailable")

        monkeypatch.setattr(bmod, "copy_artifact_tree", _fail_copy)
        with _ctx():
            await backend.execute(_request(), AsyncMock(), AsyncMock())

        assert source.exists()

    async def test_environment_config_threaded_into_trial(self, monkeypatch):
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
