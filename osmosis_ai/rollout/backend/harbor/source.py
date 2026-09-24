"""Run original Harbor tasks against images published by source image builds."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import tempfile
import tomllib
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from osmosis_ai.harbor_images import (
    TaskSource,
    bind_task_images,
    fetch_source,
    materialize_source_tasks,
    task_identities,
)
from osmosis_ai.rollout.backend.harbor.backend import HarborBackend
from osmosis_ai.rollout.types.harbor import HarborGatewayConfig


class RegistryResolver:
    def __init__(self, repository: str, password: str | None = None):
        if not re.fullmatch(
            r"[a-z0-9-]+-docker\.pkg\.dev/[a-z0-9.-]+/[a-z0-9-]+", repository
        ):
            raise ValueError("Source images require a managed GAR repository")
        self.repository = repository
        self.password = password
        self._cache: dict[str, str] = {}

    def resolve(self, tag: str) -> str:
        if tag in self._cache:
            return self._cache[tag]
        host, path = self.repository.split("/", 1)
        config = (
            Path(os.environ.get("DOCKER_CONFIG", str(Path.home() / ".docker")))
            / "config.json"
        )
        auth = (
            httpx.BasicAuth("oauth2accesstoken", self.password)
            if self.password
            else None
        )
        if auth is None and config.is_file():
            try:
                credentials = json.loads(config.read_text())
                if credentials.get("credHelpers", {}).get(host) or credentials.get(
                    "credsStore"
                ):
                    raise RuntimeError(
                        "Docker credential helpers are unsupported; start a managed source gateway to obtain registry credentials"
                    )
                entry = credentials.get("auths", {}).get(host, {})
                if entry.get("auth"):
                    username, password = (
                        base64.b64decode(entry["auth"], validate=True)
                        .decode()
                        .split(":", 1)
                    )
                    auth = httpx.BasicAuth(username, password)
            except (OSError, ValueError, TypeError, AttributeError):
                raise RuntimeError(
                    "Registry credentials could not be read; refresh Docker login or use a managed source gateway"
                ) from None
        # Read the refreshed Docker credential on each cache miss. Never follow
        # redirects or a registry-provided authentication URL with credentials.
        with httpx.Client(auth=auth, timeout=30, follow_redirects=False) as client:
            response = client.get(
                f"https://{host}/v2/{path}/environment/manifests/{tag}",
                headers={
                    "Accept": ", ".join(
                        (
                            "application/vnd.oci.image.manifest.v1+json",
                            "application/vnd.docker.distribution.manifest.v2+json",
                            "application/vnd.oci.image.index.v1+json",
                            "application/vnd.docker.distribution.manifest.list.v2+json",
                        )
                    )
                },
            )
        if response.status_code in (401, 403):
            raise RuntimeError(
                "Registry authentication failed; refresh Docker login or start a new managed source gateway"
            )
        if response.status_code != 200:
            raise RuntimeError(
                f"Source image is unavailable (GAR HTTP {response.status_code}); run 'osmosis images build' for this source first"
            )
        digest = "sha256:" + hashlib.sha256(response.content).hexdigest()
        if response.headers.get("docker-content-digest") != digest:
            raise ValueError("GAR manifest digest verification failed")
        image = f"{self.repository}/environment@{digest}"
        self._cache[tag] = image
        return image


class SourceHarborBackend(HarborBackend):
    """A pinned source has one immutable image binding per environment."""

    def __init__(
        self,
        *,
        image_bindings: dict[str, dict[str, str]],
        environment_healthcheck: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        from harbor.models.task.config import HealthcheckConfig

        super().__init__(**kwargs)
        self.image_bindings = image_bindings
        self._environment_healthcheck: HealthcheckConfig | None = (
            HealthcheckConfig.model_validate(environment_healthcheck)
            if environment_healthcheck is not None
            else None
        )
        if self.sdk_requirements:
            raise ValueError(
                "Source images cannot change their build context during rollout"
            )

    def materialize_task(
        self, task: Any, rollout_id: str, container_input: Any
    ) -> Path:
        original = Path(task.path).resolve()
        name = original.relative_to(self.tasks_dir.resolve()).as_posix()
        if name not in self.image_bindings:
            raise ValueError("Task is outside this gateway's pinned source")
        directory = super().materialize_task(task, rollout_id, container_input)
        bind_task_images(directory, self.image_bindings[name])
        if self._environment_healthcheck is not None:
            import toml

            path = directory / "task.toml"
            raw = tomllib.loads(path.read_text())
            required = self._environment_healthcheck.model_dump()
            original = raw["environment"].get("healthcheck")
            if original:
                from harbor.models.task.config import HealthcheckConfig

                original = HealthcheckConfig.model_validate(original).model_dump()
                required["command"] += " && (" + original["command"] + ")"
                required["timeout_sec"] += original["timeout_sec"]
                for key in (
                    "interval_sec",
                    "start_period_sec",
                    "start_interval_sec",
                    "retries",
                ):
                    required[key] = max(required[key], original[key])
            raw["environment"]["healthcheck"] = required
            path.write_text(toml.dumps(raw))
        return directory


def main() -> None:
    import uvicorn
    from harbor.models.environment_type import EnvironmentType
    from harbor.models.trial.config import EnvironmentConfig
    from harbor.trial.queue import TrialQueue

    from osmosis_ai.rollout.server import create_rollout_server

    config = HarborGatewayConfig.model_validate_json(
        os.environ.pop("_OSMOSIS_HARBOR_CONFIG", "{}")
    )
    environment_kwargs = dict(config.environment_kwargs)
    domain = environment_kwargs.get("domain") or os.environ.get(
        "OPENSANDBOX_DOMAIN", ""
    )
    scheme = urlsplit(domain).scheme
    # OpenSandbox's execution adapters use protocol separately from domain;
    # an explicit http:// service URL only overrides the control-plane client.
    environment_kwargs.setdefault(
        "protocol", scheme if scheme in {"http", "https"} else "https"
    )
    environment_kwargs.setdefault("use_server_proxy", True)
    source = TaskSource(**json.loads(os.environ["_OSMOSIS_HARBOR_TASK_SOURCE"]))
    published_repository = os.environ["_OSMOSIS_HARBOR_REGISTRY"]
    registry_root = published_repository.rsplit("/", 1)[0]
    registry = f"{registry_root}/{source.repository_id(os.environ['_OSMOSIS_ORGANIZATION_ID'])}"
    if published_repository != registry:
        raise ValueError("Registry namespace does not match the pinned task source")
    resolver = RegistryResolver(
        registry, os.environ.pop("_OSMOSIS_HARBOR_REGISTRY_PASSWORD", None)
    )
    with tempfile.TemporaryDirectory(prefix="harbor-source-") as directory:
        repository = Path(directory) / "repository"
        fetch_source(source, os.environ.pop("_OSMOSIS_GITHUB_TOKEN", ""), repository)
        tasks_dir = Path(directory) / "tasks"
        names = asyncio.run(
            materialize_source_tasks(repository, source.path, tasks_dir)
        )
        bindings = {
            name: {
                role: resolver.resolve(identity.tag)
                for role, identity in task_identities(tasks_dir / name).items()
            }
            for name in names
        }
        resolver.password = None
        backend = SourceHarborBackend(
            image_bindings=bindings,
            orchestrator=TrialQueue(n_concurrent=config.concurrency),
            tasks_dir=tasks_dir,
            task_mode="dataset",
            agent=config.agent,
            native_agent_kwargs=config.native_agent_kwargs,
            environment_config=EnvironmentConfig(
                type=EnvironmentType.OPENSANDBOX,
                kwargs=environment_kwargs,
            ),
            environment_healthcheck=(
                config.environment_healthcheck.model_dump()
                if config.environment_healthcheck is not None
                else None
            ),
            cleanup_successful_trials=config.cleanup_successful_trials,
        )
        representatives = list(
            {bindings[name]["environment"]: name for name in names}.values()
        )
        app = create_rollout_server(
            backend=backend, lifespan=backend.prewarm_lifespan(representatives)
        )
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=int(os.environ.get("_OSMOSIS_ROLLOUT_PORT", "8000")),
        )


if __name__ == "__main__":
    main()
