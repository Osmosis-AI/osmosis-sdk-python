"""Harbor environment placement and runtime-specific host adjustments.

Shared by both Harbor backends, so neither depends on the other.
"""

from __future__ import annotations

import logging
import os
import platform
from pathlib import Path

from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import NetworkMode
from harbor.models.task.config import TaskConfig as TaskNetworkConfig
from harbor.models.trial.config import (
    AgentConfig as HarborAgentConfig,
)
from harbor.models.trial.config import (
    EnvironmentConfig as HarborEnvironmentConfig,
)
from harbor.trial.network_policy import resolve_agent_phase_policy

logger: logging.Logger = logging.getLogger(__name__)

SKYPILOT_CONTEXT_ENV = "HARBOR_SKYPILOT_CONTEXT"


def uses_local_docker_runtime(environment_config: HarborEnvironmentConfig) -> bool:
    """Return whether Harbor will run the trial on the host Docker runtime."""
    return environment_config.type == EnvironmentType.DOCKER


def apply_managed_skypilot_placement(
    environment_config: HarborEnvironmentConfig,
) -> HarborEnvironmentConfig:
    """Resolve the SkyPilot cluster context from the run environment.

    Harbor reads its registry from ``HARBOR_SKYPILOT_REGISTRY`` but accepts the
    cluster context only as a constructor argument. Bridging the two here lets a
    rollout select ``EnvironmentType.SKYPILOT`` without naming a cluster. An
    explicit ``context_name`` takes precedence.
    """
    if environment_config.type != EnvironmentType.SKYPILOT:
        return environment_config
    if environment_config.kwargs.get("context_name"):
        return environment_config
    context_name = os.environ.get(SKYPILOT_CONTEXT_ENV)
    if context_name:
        environment_config.kwargs["context_name"] = context_name
    return environment_config


def is_loopback_url(url: str) -> bool:
    """Whether the URL's host is reachable only from this machine."""
    if not url:
        return False
    from urllib.parse import urlparse

    hostname = urlparse(url).hostname
    if hostname is None:
        return False
    if hostname == "localhost":
        return True
    import ipaddress

    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def sandbox_reaches_host_loopback(
    environment_config: HarborEnvironmentConfig,
) -> bool:
    """Whether this trial runtime can dial the host's loopback interface.

    An explicit allowlist, not "everything but cloud": every environment type
    not named here gets the loopback guard, because a silently unreachable
    chat endpoint hangs the rollout until its timeout.

    * ``SINGULARITY`` shares the host network namespace, so 127.0.0.1 works.
    * ``DOCKER`` works only through the macOS ``host.docker.internal``
      rewrite; on Linux the rewrite is a no-op and loopback is unreachable
      from a bridge network.
    * ``type is None`` is an ``import_path`` custom environment — unknown
      topology, most plausibly a bespoke local runner, so it is not blocked.
    """
    env_type = environment_config.type
    if env_type is None:
        return True
    if env_type == EnvironmentType.SINGULARITY:
        return True
    if env_type == EnvironmentType.DOCKER:
        return platform.system() == "Darwin"
    return False


def rewrite_url_for_docker(url: str) -> str:
    if platform.system() != "Darwin":
        return url
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(url)
    if parsed.hostname in ("localhost", "127.0.0.1"):
        parsed = parsed._replace(
            netloc=parsed.netloc.replace(parsed.hostname, "host.docker.internal")
        )
    return urlunparse(parsed)


def chat_endpoint_host(url: str) -> str | None:
    """Hostname the sandbox must reach for chat completions, if any.

    Harbor allowlist entries are bare hostnames or IPs -- never URLs, ports, or
    paths -- so the host is extracted rather than passed through.
    """
    if not url:
        return None
    from urllib.parse import urlparse

    hostname = urlparse(url).hostname
    return hostname or None


def load_task_network_config(task_dir: Path) -> TaskNetworkConfig | None:
    """Parse a materialized task's ``task.toml``, or ``None`` when unreadable.

    Only egress decisions read this, and the safe answer to "which network mode
    is this task in?" when the file cannot be parsed is "not allowlist" -- Harbor
    is about to fail the trial on the same file anyway, and guessing would risk
    widening a policy we could not read.
    """
    config_path = task_dir / "task.toml"
    try:
        return TaskNetworkConfig.model_validate_toml(config_path.read_text())
    except Exception:
        logger.debug("could not read %s for egress resolution", config_path)
        return None


def apply_chat_endpoint_egress(
    environment_config: HarborEnvironmentConfig,
    agent_config: HarborAgentConfig,
    *,
    task_config: TaskNetworkConfig | None,
    chat_completions_url: str,
) -> str | None:
    """Allow the sandbox to reach the chat endpoint under ``allowlist`` egress.

    A task running ``allowlist`` egress blocks the chat endpoint unless the
    host is listed; a phase whose author declared ``no-network`` is never
    modified. The two extras surfaces are gated separately: environment
    extras follow the ``[environment]`` baseline (which the verifier also
    inherits — its own ``resolve_baseline``, deliberately not
    ``resolve_agent_env_baseline``, whose extras merge could upgrade a
    no-network baseline), and agent extras follow the phase policy Harbor
    itself resolves per step, skipped entirely if any step is offline.

    Returns the host that was allowed, or ``None`` when nothing was needed.
    """
    host = chat_endpoint_host(chat_completions_url)
    if host is None or task_config is None:
        return None

    baseline = task_config.environment.resolve_baseline()
    # Both call sites hand over freshly built configs, so extra_allowed_hosts
    # is empty here and the resolver's extras merge is a no-op.
    agent_modes = {
        resolve_agent_phase_policy(
            task_config, agent_config, baseline, step_cfg=step
        ).network_mode
        for step in (task_config.steps or [None])
    }
    surfaces: list[list[str]] = []
    if baseline.network_mode is NetworkMode.ALLOWLIST:
        surfaces.append(environment_config.extra_allowed_hosts)
    if (
        NetworkMode.ALLOWLIST in agent_modes
        and NetworkMode.NO_NETWORK not in agent_modes
    ):
        surfaces.append(agent_config.extra_allowed_hosts)
    if not surfaces:
        return None

    for hosts in surfaces:
        if host not in hosts:
            hosts.append(host)
    return host
