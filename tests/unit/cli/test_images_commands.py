import hashlib
import io
import json
import tarfile
import urllib.error
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


@pytest.mark.parametrize("method", ["submit_image_build", "get_image_build"])
def test_api_propagates_image_request_timeout(monkeypatch, method):
    request = Mock(return_value={"phase": "queued"})
    monkeypatch.setattr("osmosis_ai.platform.api.client.platform_request", request)
    kwargs = (
        {"repository": "https://github.com/acme/job"}
        if method == "submit_image_build"
        else {"git_identity": "acme/job"}
    )
    getattr(OsmosisClient(), method)(request_id="request", timeout=0.25, **kwargs)
    assert request.call_args.kwargs["timeout"] == 0.25


def test_pull_credentials_client_encodes_request_id_and_scopes_image(monkeypatch):
    request = Mock(return_value={"registry": "registry.test"})
    monkeypatch.setattr("osmosis_ai.platform.api.client.platform_request", request)
    result = OsmosisClient().get_image_pull_credentials(
        "../other?query",
        image="registry/image@sha256:" + "a" * 64,
        git_identity="acme/job",
    )
    assert result == {"registry": "registry.test"}
    request.assert_called_once_with(
        "/api/cli/image-builds/..%2Fother%3Fquery/pull-credentials",
        method="POST",
        data={"image": "registry/image@sha256:" + "a" * 64},
        credentials=None,
        git_identity="acme/job",
    )


def test_info_normalizes_repository_and_returns_build_progress(monkeypatch):
    client = Mock()
    client.get_image_build.return_value = {
        "phase": "building",
        "source_revision": "a" * 40,
        "task_count": 4,
        "image_count": 3,
        "completed_image_count": 2,
    }
    monkeypatch.setattr(images, "OsmosisClient", lambda: client)
    result = images.info(repository="git@github.com:Acme/Job.git", request_id="request")
    client.get_image_build.assert_called_once_with("request", git_identity="acme/job")
    assert result.data == {
        "request_id": "request",
        **client.get_image_build.return_value,
    }


@pytest.mark.parametrize(
    "invalid",
    [
        {"ref": ""},
        {"ref": "a" * 256},
        {"ref": "main\n"},
        {"ref": "main\x7f"},
        {"tasks_dir": ".."},
        {"tasks_dir": "../tasks"},
        {"tasks_dir": "/tasks"},
        {"tasks_dir": "tasks//one"},
        {"tasks_dir": "tasks\\one"},
        {"tasks_dir": "tasks\0"},
        {"request_id": "not-a-uuid"},
    ],
)
def test_invalid_request_inputs_fail_before_persisting_state(tmp_path, invalid):
    kwargs = {
        "repository": "https://github.com/acme/job",
        "ref": "main",
        "tasks_dir": "tasks",
        **invalid,
    }
    with pytest.raises(CLIError) as error:
        images.prepare_request(tmp_path, **kwargs)
    assert error.value.code == "VALIDATION"
    assert not (tmp_path / "request.json").exists()


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
    for call in client.submit_image_build.call_args_list:
        assert {k: v for k, v in call.kwargs.items() if k != "timeout"} == saved[
            "request"
        ]
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
    client.reset_mock()
    clock = Mock(side_effect=[0, 1, 31, 31])
    monkeypatch.setattr(images.time, "monotonic", clock)
    monkeypatch.setattr(images.time, "sleep", lambda _: None)
    with pytest.raises(CLIError, match="cloud builds continue"):
        images.wait_for_build(client, saved, tmp_path, 30, True)
    client.submit_image_build.assert_called_once_with(**saved["request"], timeout=29)
    client.get_image_build.assert_not_called()
    assert json.loads((tmp_path / "status.json").read_text())["phase"] == "queued"
    clock.side_effect = [40, 41]
    client.submit_image_build.return_value = {"phase": "completed"}
    resumed = state(tmp_path)
    assert resumed["request_id"] == saved["request_id"]
    assert (
        images.wait_for_build(client, resumed, tmp_path, 30, True)["phase"]
        == "completed"
    )
    assert len(client.method_calls) == 2
    assert client.method_calls[0] == client.method_calls[1]


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409, 422])
def test_nonretryable_image_api_error_propagates(tmp_path, monkeypatch, status):
    client = Mock()
    error = PlatformAPIError("rejected", status_code=status)
    client.submit_image_build.side_effect = error
    sleep = Mock()
    monkeypatch.setattr(images.time, "sleep", sleep)
    with pytest.raises(PlatformAPIError) as raised:
        images.wait_for_build(client, state(tmp_path), tmp_path, 30, True)
    assert raised.value is error
    client.submit_image_build.assert_called_once()
    client.get_image_build.assert_not_called()
    sleep.assert_not_called()


@pytest.mark.parametrize(
    "error", [urllib.error.URLError("offline"), TimeoutError(), ConnectionError()]
)
def test_transport_failure_retries_same_submission(tmp_path, monkeypatch, error):
    saved = state(tmp_path)
    client = Mock()
    client.submit_image_build.side_effect = [error, {"phase": "completed"}]
    monkeypatch.setattr(images.time, "sleep", lambda _: None)
    assert (
        images.wait_for_build(client, saved, tmp_path, 30, True)["phase"] == "completed"
    )
    assert client.submit_image_build.call_count == 2
    assert all(
        call.kwargs["request_id"] == saved["request_id"]
        for call in client.submit_image_build.call_args_list
    )


def test_submit_and_poll_receive_remaining_local_wait_time(tmp_path, monkeypatch):
    now = [0.0]
    monkeypatch.setattr(images.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(
        images.time, "sleep", lambda seconds: now.__setitem__(0, now[0] + seconds)
    )
    client = Mock()

    def submit(**kwargs):
        assert kwargs["timeout"] == 12
        now[0] += 0.5
        return {"phase": "building"}

    def poll(*args, **kwargs):
        assert kwargs["timeout"] == 1.5
        now[0] += kwargs["timeout"]
        raise TimeoutError()

    client.submit_image_build.side_effect = submit
    client.get_image_build.side_effect = poll
    with pytest.raises(CLIError, match="Local wait limit"):
        images.wait_for_build(client, state(tmp_path), tmp_path, 12, True)
    assert now[0] == 12
    client.get_image_build.assert_called_once()


def fixture_artifacts(tmp_path, corruption=None):
    image = "registry/task@sha256:" + "a" * 64
    buffer = io.BytesIO()
    config = f'[environment]\ndocker_image = "{image}"\n[verifier.environment]\ndocker_image = "{image}"\n'.encode()
    if corruption == "binding":
        config = config.replace(b"registry/task", b"registry/other")
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        member = tarfile.TarInfo(
            "outside/task.toml"
            if corruption == "outside-directory"
            else "tasks/group/one/task.toml"
        )
        member.size = len(config)
        archive.addfile(member, io.BytesIO(config))
        if corruption == "traversal":
            archive.addfile(tarfile.TarInfo("../../outside"))
        if corruption in ("unlisted-task", "nested-fixture"):
            member = tarfile.TarInfo(
                "tasks/group/two/task.toml"
                if corruption == "unlisted-task"
                else "tasks/group/one/environment/fixture/task.toml"
            )
            member.size = len(config)
            archive.addfile(member, io.BytesIO(config))
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
    if corruption == "missing-key":
        manifest["tasks"][0]["environments"]["environment"] = "unknown-image"
    if corruption == "outside-directory":
        manifest["tasks"][0]["task"] = "tasks/../outside"
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
    "corruption",
    [
        None,
        "revision",
        "binding",
        "roles",
        "missing-key",
        "traversal",
        "outside-directory",
        "unlisted-task",
    ],
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
        with pytest.raises(CLIError) as error:
            images.collect(client, saved, status, tmp_path)
        assert error.value.code == "PLATFORM_ERROR"
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


def test_nested_codebase_fixture_is_not_an_additional_task(tmp_path, monkeypatch):
    saved = state(tmp_path)
    status, access, files = fixture_artifacts(tmp_path, "nested-fixture")
    client = Mock()
    client.get_image_build_artifacts.return_value = access
    monkeypatch.setattr(
        images,
        "download",
        lambda reference, capability, path: path.write_bytes(files[reference["name"]]),
    )
    assert set(images.collect(client, saved, status, tmp_path)["tasks"]) == {
        "group/one"
    }


@pytest.mark.parametrize("dangling", [False, True])
def test_resume_rejects_root_task_symlink(tmp_path, monkeypatch, dangling):
    saved = state(tmp_path)
    status, access, files = fixture_artifacts(tmp_path)
    client = Mock()
    client.get_image_build_artifacts.return_value = access
    monkeypatch.setattr(
        images,
        "download",
        lambda reference, capability, path: path.write_bytes(files[reference["name"]]),
    )
    images.collect(client, saved, status, tmp_path)
    external = tmp_path / "external"
    (tmp_path / "tasks").rename(external)
    destination = tmp_path / "tasks"
    destination.symlink_to(
        tmp_path / "missing" if dangling else external, target_is_directory=True
    )
    before = (external / "group/one/task.toml").read_bytes()
    with pytest.raises(CLIError, match="must not be a symlink") as error:
        images.collect(client, saved, status, tmp_path)
    assert error.value.code == "CONFLICT"
    assert destination.is_symlink()
    assert (external / "group/one/task.toml").read_bytes() == before


def test_atomic_state_write_does_not_follow_existing_symlinks(tmp_path):
    victim = tmp_path / "victim"
    victim.write_text("preserve")
    destination = tmp_path / "request.json"
    destination.symlink_to(victim)
    predictable_temporary = tmp_path / "request.tmp"
    predictable_temporary.symlink_to(victim)
    images.write_json(destination, {"request_id": "new"})
    assert victim.read_text() == "preserve"
    assert not destination.is_symlink()
    assert json.loads(destination.read_text()) == {"request_id": "new"}
    assert destination.stat().st_mode & 0o777 == 0o600
    assert {p.name for p in tmp_path.iterdir()} == {
        "victim",
        "request.json",
        "request.tmp",
    }


@pytest.mark.parametrize("valid_checksum", [True, False])
def test_atomic_download_preserves_symlink_target_and_failed_transfer(
    tmp_path, monkeypatch, valid_checksum
):
    content = b"new artifact"
    victim = tmp_path / "victim"
    victim.write_bytes(b"preserve")
    destination = tmp_path / "manifest.json"
    destination.symlink_to(victim)
    opener = Mock()
    opener.open.return_value = io.BytesIO(content)
    monkeypatch.setattr(images.urllib.request, "build_opener", lambda *args: opener)
    reference = {
        "sha256": hashlib.sha256(content).hexdigest() if valid_checksum else "bad"
    }
    capability = {
        "reference": reference,
        "url": "https://storage.googleapis.com/artifact",
    }
    if valid_checksum:
        images.download(reference, capability, destination)
        assert destination.read_bytes() == content
        assert not destination.is_symlink()
    else:
        with pytest.raises(CLIError, match="checksum"):
            images.download(reference, capability, destination)
        assert destination.is_symlink()
    assert victim.read_bytes() == b"preserve"
    assert {p.name for p in tmp_path.iterdir()} == {"victim", "manifest.json"}


@pytest.mark.parametrize(
    "url",
    [
        "https://[invalid/path?secret",
        "https://storage.googleapis.com:bad/path?secret",
        "https://storage.googleapis.com:65536/path?secret",
    ],
)
def test_malformed_download_origin_has_safe_platform_error(tmp_path, url):
    with pytest.raises(
        CLIError, match="Unexpected image artifact download origin"
    ) as error:
        images.download({}, {"reference": {}, "url": url}, tmp_path / "artifact")
    assert error.value.code == "PLATFORM_ERROR"
    assert "secret" not in str(error.value)
    assert not (tmp_path / "artifact").exists()


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
