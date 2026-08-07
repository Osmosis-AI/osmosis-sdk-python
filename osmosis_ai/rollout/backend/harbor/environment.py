"""Harbor environment placement and runtime-specific host adjustments.

Shared by both Harbor backends, so neither depends on the other.
"""

from __future__ import annotations

import os
import platform

from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import (
    EnvironmentConfig as HarborEnvironmentConfig,
)

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
