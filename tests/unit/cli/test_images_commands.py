import hashlib
import io
import json
import tarfile
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.auth.platform_client import PlatformAPIError
from osmosis_ai.platform.cli import images


@pytest.fixture(autouse=True)
def platform(monkeypatch):
    monkeypatch.setattr(images, "get_platform_url", lambda: "https://platform.test")


def state(tmp_path):
    return images.prepare_request(
        tmp_path, "git@github.com:Acme/Job.git", "main", "tasks"
    )


def test_api_sends_repository_not_a_task_inventory(monkeypatch):
    request = Mock(return_value={"phase": "queued"})
    monkeypatch.setattr("osmosis_ai.platform.api.client.platform_request", request)
    OsmosisClient().submit_image_build(
        repository="git@github.com:Acme/Job.git", request_id="request"
    )
    assert request.call_args.args == ("/api/cli/image-builds",)
    assert request.call_args.kwargs["data"] == {
        "request_id": "request",
        "repository": "https://github.com/acme/job",
        "ref": "HEAD",
        "tasks_dir": "tasks",
    }
    assert request.call_args.kwargs["git_identity"] == "acme/job"


def test_retry_retains_id_and_rejects_different_repo_ref_or_platform(
    tmp_path, monkeypatch
):
    original = state(tmp_path)
    assert state(tmp_path) == original
    assert (tmp_path / "request.json").stat().st_mode & 0o777 == 0o600
    assert list(tmp_path.iterdir()) == [tmp_path / "request.json"]
    for repository, ref in [
        ("https://github.com/acme/other", "main"),
        ("https://github.com/acme/job", "other"),
    ]:
        with pytest.raises(CLIError, match="inputs changed"):
            images.prepare_request(tmp_path, repository, ref, "tasks")
    monkeypatch.setattr(images, "get_platform_url", lambda: "https://other.test")
    with pytest.raises(CLIError, match="inputs changed"):
        state(tmp_path)


def test_lost_submission_reply_and_poll_failure_resume_same_id(tmp_path, monkeypatch):
    saved = state(tmp_path)
    client = Mock()
    client.submit_image_build.side_effect = [
        PlatformAPIError("lost reply", status_code=503),
        {"phase": "building"},
    ]
    client.get_image_build.side_effect = [
        PlatformAPIError("outage", status_code=503),
        {"phase": "completed"},
    ]
    monkeypatch.setattr(images.time, "sleep", lambda _: None)
    assert (
        images.wait_for_build(client, saved, tmp_path, 30, True)["phase"] == "completed"
    )
    assert (
        client.submit_image_build.call_args_list[0]
        == client.submit_image_build.call_args_list[1]
    )
    assert client.get_image_build.call_args.args == (saved["request_id"],)


def test_timeout_preserves_remote_job_and_no_wait_returns_immediately(
    tmp_path, monkeypatch
):
    client = Mock()
    client.submit_image_build.return_value = {"phase": "queued"}
    saved = state(tmp_path)
    assert (
        images.wait_for_build(client, saved, tmp_path, 30, False)["phase"] == "queued"
    )
    monkeypatch.setattr(images.time, "monotonic", Mock(side_effect=[0, 31]))
    with pytest.raises(CLIError, match="cloud builds continue"):
        images.wait_for_build(client, saved, tmp_path, 30, True)
    assert client.method_calls == [client.method_calls[0]]


def fixture_artifacts(tmp_path, corruption=None):
    image = "registry/task@sha256:" + "a" * 64
    buffer = io.BytesIO()
    config = f'[environment]\ndocker_image = "{image}"\n[verifier.environment]\ndocker_image = "{image}"\n'.encode()
    if corruption == "binding":
        config = config.replace(b"registry/task", b"registry/other")
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo("tasks/group/one/task.toml")
        member.size = len(config)
        archive.addfile(member, io.BytesIO(config))
        if corruption == "traversal":
            archive.addfile(tarfile.TarInfo("../../outside"))
    bundle_bytes = buffer.getvalue()

    def reference(name, data):
        return {
            "bucket": "bucket",
            "name": name,
            "generation": 1,
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    bundle = reference("bundle", bundle_bytes)
    manifest = {
        "source_revision": "a" * 40,
        "bundle": bundle,
        "images": [{"key": "key", "image": image}],
        "named_images": {"trainer": image},
        "tasks": [
            {
                "task": "tasks/group/one",
                "environments": {"environment": "key", "verifier.environment": "key"},
            }
        ],
    }
    if corruption == "roles":
        manifest["tasks"][0]["environments"].pop("verifier.environment")
    manifest_bytes = json.dumps(manifest).encode()
    ref = reference("manifest", manifest_bytes)
    status = {
        "phase": "completed",
        "source_revision": "b" * 40 if corruption == "revision" else "a" * 40,
        "result": {"manifest": ref, "bundle": bundle, "task_count": 1},
    }
    access = {
        "manifest": {
            "reference": ref,
            "url": "https://storage.googleapis.com/manifest?secret",
        },
        "bundle": {
            "reference": bundle,
            "url": "https://storage.googleapis.com/bundle?secret",
        },
    }
    return status, access, {"manifest": manifest_bytes, "bundle": bundle_bytes}


@pytest.mark.parametrize(
    "corruption", [None, "revision", "binding", "roles", "traversal"]
)
def test_download_validates_commit_task_roles_and_preserves_local_edits(
    tmp_path, monkeypatch, corruption
):
    saved = state(tmp_path)
    status, access, files = fixture_artifacts(tmp_path, corruption)
    client = Mock()
    client.get_image_build_artifacts.return_value = access

    def download(reference, capability, path):
        assert capability["reference"] == reference
        path.write_bytes(files[reference["name"]])

    monkeypatch.setattr(images, "download", download)
    if corruption:
        with pytest.raises((CLIError, tarfile.TarError)):
            images.collect(client, saved, status, tmp_path)
        assert not (tmp_path / "images.json").exists()
        return
    summary = images.collect(client, saved, status, tmp_path)
    assert set(summary["tasks"]) == {"group/one"}
    assert summary["named_images"]["trainer"].startswith("registry/task@sha256:")
    assert images.collect(client, saved, status, tmp_path) == summary
    (tmp_path / "tasks/group/one/task.toml").write_text("changed locally")
    with pytest.raises(CLIError, match="changed locally"):
        images.collect(client, saved, status, tmp_path)
    assert "secret" not in (tmp_path / "request.json").read_text()


def test_cli_repo_only_json_submission(tmp_path, monkeypatch):
    from osmosis_ai.cli.main import _register_commands, app

    _register_commands()
    client = Mock()
    client.submit_image_build.return_value = {"phase": "queued"}
    monkeypatch.setattr(images, "OsmosisClient", lambda: client)
    result = CliRunner().invoke(
        app,
        [
            "--json",
            "images",
            "build",
            "--repo",
            "https://github.com/acme/job",
            "--no-wait",
            "--output-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)
    body = client.submit_image_build.call_args.kwargs
    assert body["tasks_dir"] == "tasks" and body["ref"] == "HEAD"
    assert "tasks" not in body


def test_download_uses_signed_access_without_platform_auth_and_checks_bytes(
    tmp_path, monkeypatch
):
    content = b"published artifact"
    reference = {"sha256": hashlib.sha256(content).hexdigest()}
    url = "https://storage.googleapis.com/bucket/object?temporary-access"
    opener = Mock()
    opener.open.return_value = io.BytesIO(content)
    monkeypatch.setattr(images.urllib.request, "build_opener", lambda *args: opener)
    destination = tmp_path / "artifact"
    images.download(reference, {"reference": reference, "url": url}, destination)
    opener.open.assert_called_once_with(url, timeout=300)
    assert destination.read_bytes() == content
    with pytest.raises(CLIError, match="origin"):
        images.download(
            reference,
            {"reference": reference, "url": "https://untrusted.test/object"},
            destination,
        )
    wrong = {"sha256": "a" * 64}
    opener.open.return_value = io.BytesIO(content)
    with pytest.raises(CLIError, match="checksum"):
        images.download(wrong, {"reference": wrong, "url": url}, destination)
