"""Sandbox egress to the chat endpoint under Harbor's network policies.

The container needs exactly one thing: outbound access to the chat-endpoint
host. Harbor defaults to ``public``, where nothing is needed; a task running
``allowlist`` blocks the chat endpoint unless the host is listed, and a phase
that declares ``no-network`` is never modified.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("harbor")

from harbor.models.trial.config import AgentConfig, EnvironmentConfig

from osmosis_ai.rollout.backend.harbor.environment import (
    apply_chat_endpoint_egress,
    chat_endpoint_host,
    load_task_network_config,
)

CHAT_URL = "https://chat.example.com/v1/rollouts/abc123"

_TASK_TOML = """\
version = "1.0"

[verifier]
timeout_sec = 60.0

[agent]
timeout_sec = 120.0

[environment]
build_timeout_sec = 600.0
cpus = 1
memory_mb = 512
storage_mb = 1024
"""


def _task_dir(tmp_path: Path, *, environment: str = "", agent: str = "") -> Path:
    task_dir = tmp_path / "task"
    task_dir.mkdir(parents=True, exist_ok=True)
    body = _TASK_TOML
    if environment:
        body = body.replace(
            "[environment]\nbuild_timeout_sec",
            f"[environment]\n{environment}\nbuild_timeout_sec",
        )
    if agent:
        body = body.replace("[agent]\ntimeout_sec", f"[agent]\n{agent}\ntimeout_sec")
    (task_dir / "task.toml").write_text(body, encoding="utf-8")
    return task_dir


def _apply(
    task_dir: Path, url: str = CHAT_URL
) -> tuple[str | None, EnvironmentConfig, AgentConfig]:
    environment = EnvironmentConfig()
    agent = AgentConfig(import_path="x:Y")
    allowed = apply_chat_endpoint_egress(
        environment,
        agent,
        task_config=load_task_network_config(task_dir),
        chat_completions_url=url,
    )
    return allowed, environment, agent


# --------------------------------------------------------------------------- #
# Host extraction
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://chat.example.com/v1/rollouts/a", "chat.example.com"),
        ("http://host.docker.internal:8791/v1/rollouts/a", "host.docker.internal"),
        ("http://127.0.0.1:9000/v1", "127.0.0.1"),
        ("", None),
        ("not a url", None),
    ],
)
def test_chat_endpoint_host(url: str, expected: str | None) -> None:
    # Harbor allowlist entries are bare hosts: no scheme, port, or path.
    assert chat_endpoint_host(url) == expected


# --------------------------------------------------------------------------- #
# Policy-aware injection
# --------------------------------------------------------------------------- #


def test_an_explicit_agent_allowlist_also_gets_the_host(tmp_path: Path) -> None:
    task_dir = _task_dir(tmp_path, agent='network_mode = "allowlist"')
    allowed, _environment, agent = _apply(task_dir)
    assert allowed == "chat.example.com"
    assert agent.extra_allowed_hosts == ["chat.example.com"]


def test_public_egress_needs_nothing(tmp_path: Path) -> None:
    # Harbor's default. Adding extras here is a no-op that only emits a warning.
    allowed, environment, agent = _apply(_task_dir(tmp_path))
    assert allowed is None
    assert environment.extra_allowed_hosts == []
    assert agent.extra_allowed_hosts == []


def test_no_network_is_never_silently_upgraded(tmp_path: Path) -> None:
    # merge_extra_allowlists turns any non-public policy into an allowlist, so
    # injecting here would hand network access to a task that asked for none.
    task_dir = _task_dir(tmp_path, environment='network_mode = "no-network"')
    allowed, environment, agent = _apply(task_dir)
    assert allowed is None
    assert environment.extra_allowed_hosts == []
    assert agent.extra_allowed_hosts == []


def test_an_unreadable_task_config_injects_nothing(tmp_path: Path) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text("not = valid = toml", encoding="utf-8")
    assert load_task_network_config(task_dir) is None
    allowed, environment, _agent = _apply(task_dir)
    assert allowed is None
    assert environment.extra_allowed_hosts == []


def test_a_missing_task_config_injects_nothing(tmp_path: Path) -> None:
    assert load_task_network_config(tmp_path / "absent") is None


def test_no_chat_url_injects_nothing(tmp_path: Path) -> None:
    task_dir = _task_dir(tmp_path, environment='network_mode = "allowlist"')
    allowed, environment, _agent = _apply(task_dir, url="")
    assert allowed is None
    assert environment.extra_allowed_hosts == []


def test_the_host_is_added_once(tmp_path: Path) -> None:
    task_dir = _task_dir(tmp_path, environment='network_mode = "allowlist"')
    environment = EnvironmentConfig(extra_allowed_hosts=["chat.example.com"])
    agent = AgentConfig(import_path="x:Y")
    apply_chat_endpoint_egress(
        environment,
        agent,
        task_config=load_task_network_config(task_dir),
        chat_completions_url=CHAT_URL,
    )
    assert environment.extra_allowed_hosts == ["chat.example.com"]


def test_existing_allowlist_entries_are_preserved(tmp_path: Path) -> None:
    task_dir = _task_dir(tmp_path, environment='network_mode = "allowlist"')
    environment = EnvironmentConfig(extra_allowed_hosts=["pypi.org"])
    agent = AgentConfig(import_path="x:Y")
    apply_chat_endpoint_egress(
        environment,
        agent,
        task_config=load_task_network_config(task_dir),
        chat_completions_url=CHAT_URL,
    )
    assert environment.extra_allowed_hosts == ["pypi.org", "chat.example.com"]


def test_a_docker_rewritten_localhost_url_is_allowlisted_by_its_rewritten_host(
    tmp_path: Path,
) -> None:
    # Development against a local stub: the URL the sandbox actually dials is
    # the rewritten one, so that is the host the allowlist must carry.
    task_dir = _task_dir(tmp_path, environment='network_mode = "allowlist"')
    allowed, environment, _agent = _apply(
        task_dir, url="http://host.docker.internal:8791/v1/rollouts/a"
    )
    assert allowed == "host.docker.internal"
    assert environment.extra_allowed_hosts == ["host.docker.internal"]


# --------------------------------------------------------------------------- #
# Mixed-mode tasks: the two extras surfaces resolve independently
# --------------------------------------------------------------------------- #


def _resolved(
    task_dir: Path, environment: EnvironmentConfig, agent: AgentConfig
) -> Any:
    """Ask Harbor itself what policies it will enforce, post-injection."""
    from harbor.models.task.config import TaskConfig as HarborTaskConfig
    from harbor.trial.network_policy import (
        resolve_agent_env_baseline,
        resolve_agent_phase_policy,
        resolve_verifier_env_baseline,
        resolve_verifier_phase_policy,
    )

    task_cfg = HarborTaskConfig.model_validate_toml(
        (task_dir / "task.toml").read_text()
    )
    env_baseline = resolve_agent_env_baseline(task_cfg, environment)
    verifier_baseline = resolve_verifier_env_baseline(
        task_cfg, environment, None, env_config=task_cfg.environment
    )
    return {
        "agent_env": env_baseline,
        "agent_phase": resolve_agent_phase_policy(task_cfg, agent, env_baseline),
        "verifier_phase": resolve_verifier_phase_policy(
            task_cfg, baseline=verifier_baseline
        ),
    }


def test_an_offline_agent_phase_stays_offline_under_an_allowlist_environment(
    tmp_path: Path,
) -> None:
    # [environment] allowlist for setup, [agent] no-network for an offline solve.
    # Writing agent extras here would put the agent back online, because
    # merge_extra_allowlists turns any non-public policy into an allowlist.
    task_dir = _task_dir(
        tmp_path,
        environment='network_mode = "allowlist"\nallowed_hosts = ["pypi.org"]',
        agent='network_mode = "no-network"',
    )
    allowed, environment, agent = _apply(task_dir)
    assert allowed == "chat.example.com"
    assert environment.extra_allowed_hosts == ["chat.example.com"]
    assert agent.extra_allowed_hosts == []

    policies = _resolved(task_dir, environment, agent)
    assert policies["agent_phase"].network_mode.value == "no-network"
    assert policies["agent_phase"].allowed_hosts == []


def test_an_offline_environment_does_not_leak_through_the_verifier(
    tmp_path: Path,
) -> None:
    # The verifier inherits the environment baseline, so env extras written
    # against a no-network baseline would put the grader online.
    task_dir = _task_dir(
        tmp_path,
        environment='network_mode = "no-network"',
        agent='network_mode = "allowlist"',
    )
    allowed, environment, agent = _apply(task_dir)
    assert allowed == "chat.example.com"
    assert environment.extra_allowed_hosts == []
    assert agent.extra_allowed_hosts == ["chat.example.com"]

    policies = _resolved(task_dir, environment, agent)
    assert policies["agent_env"].network_mode.value == "no-network"
    assert policies["verifier_phase"].network_mode.value == "no-network"
    assert policies["agent_phase"].network_mode.value == "allowlist"


def test_an_offline_step_keeps_the_agent_surface_untouched(tmp_path: Path) -> None:
    # agent extras apply to every step, so one offline step vetoes the surface.
    task_dir = tmp_path / "task"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.toml").write_text(
        _TASK_TOML.replace(
            "[environment]\nbuild_timeout_sec",
            '[environment]\nnetwork_mode = "allowlist"\nbuild_timeout_sec',
        )
        + '\n[[steps]]\nname = "solve"\n[steps.agent]\nnetwork_mode = "no-network"\n',
        encoding="utf-8",
    )
    allowed, environment, agent = _apply(task_dir)
    assert allowed == "chat.example.com"
    assert environment.extra_allowed_hosts == ["chat.example.com"]
    assert agent.extra_allowed_hosts == []


def test_a_public_environment_with_an_allowlist_agent_skips_env_extras(
    tmp_path: Path,
) -> None:
    # Harbor ignores extras on a public policy and warns per trial; skipping the
    # surface keeps that noise out of a 200-row run.
    task_dir = _task_dir(tmp_path, agent='network_mode = "allowlist"')
    allowed, environment, agent = _apply(task_dir)
    assert allowed == "chat.example.com"
    assert environment.extra_allowed_hosts == []
    assert agent.extra_allowed_hosts == ["chat.example.com"]


def test_the_plain_allowlist_task_still_gets_both_surfaces(tmp_path: Path) -> None:
    task_dir = _task_dir(tmp_path, environment='network_mode = "allowlist"')
    allowed, environment, agent = _apply(task_dir)
    assert allowed == "chat.example.com"
    assert environment.extra_allowed_hosts == ["chat.example.com"]
    assert agent.extra_allowed_hosts == ["chat.example.com"]

    policies = _resolved(task_dir, environment, agent)
    assert policies["agent_phase"].network_mode.value == "allowlist"
    assert "chat.example.com" in policies["agent_phase"].allowed_hosts
