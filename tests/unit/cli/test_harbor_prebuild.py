from __future__ import annotations

import json
from pathlib import Path

from osmosis_ai.cli import main as cli
from osmosis_ai.rollout.backend.harbor import images


def test_prebuild_forwards_remote_dataset_and_google_options(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    captured: dict[str, object] = {}

    async def fake_build(
        dataset: str, **kwargs: object
    ) -> images.BuildAndPublishResult:
        captured["dataset"] = dataset
        captured.update(kwargs)
        image = images.PublishedImage(
            "us-west1-docker.pkg.dev/acme/repo/harbor:osmosis--abc",
            "sha256:def",
        )
        return images.BuildAndPublishResult(
            dataset=dataset,
            dataset_path=tmp_path / "cached",
            task_count=2,
            environments=(images.PublishedEnvironment(image, "abc", ("one", "two")),),
        )

    monkeypatch.setattr(images, "build_and_publish", fake_build)

    rc = cli.main(
        [
            "--json",
            "harbor",
            "prebuild",
            "https://github.com/acme/tasks.git",
            "--image-repository",
            "us-west1-docker.pkg.dev/acme/repo/harbor",
            "--build-system",
            "google-cloud-build",
            "--gcp-project",
            "acme",
            "--gcp-region",
            "us-west1",
            "--build-arg",
            "VERSION=1",
        ]
    )

    assert rc == 0
    assert captured == {
        "dataset": "https://github.com/acme/tasks.git",
        "image_repository": "us-west1-docker.pkg.dev/acme/repo/harbor",
        "build_system": "google-cloud-build",
        "platform": "linux/amd64",
        "build_args": {"VERSION": "1"},
        "buildx_builder": None,
        "gcp_project": "acme",
        "gcp_region": "us-west1",
    }
    payload = json.loads(capsys.readouterr().out)
    assert payload["operation"] == "harbor.prebuild"
    assert payload["resource"]["task_count"] == 2
    assert payload["resource"]["images"][0]["immutable_image"].endswith("@sha256:def")


def test_prebuild_rejects_invalid_build_arg(monkeypatch, capsys) -> None:
    rc = cli.main(
        [
            "--json",
            "harbor",
            "prebuild",
            "./tasks",
            "--image-repository",
            "example.com/acme/harbor",
            "--build-arg",
            "INVALID",
        ]
    )

    assert rc == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"]["code"] == "VALIDATION"
    assert "KEY=VALUE" in payload["error"]["message"]


def test_google_cloud_build_requires_project_and_region(capsys) -> None:
    rc = cli.main(
        [
            "--json",
            "harbor",
            "prebuild",
            "https://github.com/acme/tasks.git",
            "--image-repository",
            "us-west1-docker.pkg.dev/acme/repo/harbor",
            "--build-system",
            "google-cloud-build",
        ]
    )

    assert rc == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["error"]["code"] == "VALIDATION"
    assert "gcp_project and gcp_region are required" in payload["error"]["message"]
