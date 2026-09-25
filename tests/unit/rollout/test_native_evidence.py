from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from osmosis_ai.rollout.backend.harbor.evidence import retain_trial_evidence
from osmosis_ai.rollout.utils.evidence import (
    trial_evidence_inventory,
    verify_trial_evidence,
)


def source(tmp_path):
    trial = tmp_path / "trial"
    (trial / "agent").mkdir(parents=True)
    (trial / "verifier").mkdir()
    (trial / "result.json").write_text(
        json.dumps(
            {
                "config": {
                    "api_key": "controller-secret",
                    "env": {"HF_TOKEN": "hf-hidden"},
                },
                "reward": 1.0,
                "message": "Bearer message-secret",
            }
        )
    )
    (trial / "trial.log").write_text(
        "api_key=log-secret Authorization: Bearer auth-secret\nhttps://user:url-secret@host/path\ncontroller-secret\n"
    )
    (trial / "agent/trajectory.json").write_text(
        json.dumps(
            {"steps": [{"password": "json-secret"}], "message": "TOKEN=string-secret"}
        )
    )
    (trial / "verifier/reward.txt").write_text("1.0\n")
    (trial / "config.json").write_text("not exported")
    return trial


def test_native_evidence_is_sanitized_complete_and_verifiable(tmp_path):
    trial, root = source(tmp_path), tmp_path / "out"
    assert retain_trial_evidence(trial, root, "one", api_key="controller-secret")
    destination = root / "one/harbor"
    manifest = verify_trial_evidence(destination, "one")
    assert {item["path"] for item in manifest["files"]} == {
        "result.json",
        "logs/trial.log",
        "logs/agent/trajectory.json",
        "logs/verifier/reward.txt",
    }
    retained = b"".join(
        path.read_bytes() for path in destination.rglob("*") if path.is_file()
    )
    for secret in (
        b"controller-secret",
        b"hf-hidden",
        b"message-secret",
        b"log-secret",
        b"auth-secret",
        b"url-secret",
        b"json-secret",
        b"string-secret",
    ):
        assert secret not in retained
    result = json.loads((destination / "result.json").read_text())
    assert result["reward"] == 1.0
    assert result["config"]["env"]["HF_TOKEN"] == "[REDACTED]"
    assert trial_evidence_inventory(root, ["one", "missing"])[
        "missing_rollout_ids"
    ] == ["missing"]
    assert not trial_evidence_inventory(root, ["one", "missing"])["complete"]


def test_native_json_keys_and_credential_spellings_are_sanitized(tmp_path):
    trial, root = source(tmp_path), tmp_path / "out"
    values = {
        "controller-secret": "ordinary",
        "accessToken": "access-sensitive",
        "clientSecret": "client-sensitive",
        "aws_secret_access_key": "aws-sensitive",
        "privateKey": "private-sensitive",
        "total_completion_tokens": 123,
        "message": "Authorization: Basic basic-sensitive",
    }
    (trial / "agent/trajectory.json").write_text(json.dumps(values))
    assert retain_trial_evidence(trial, root, "one", api_key="controller-secret")
    value = json.loads((root / "one/harbor/logs/agent/trajectory.json").read_text())
    serialized = json.dumps(value)
    assert "controller-secret" not in serialized
    assert "sensitive" not in serialized
    assert value["total_completion_tokens"] == 123


@pytest.mark.parametrize(
    "scheme", ["https", "postgresql", "redis", "ssh", "custom+tls"]
)
def test_url_userinfo_is_sanitized_in_native_results_and_logs(tmp_path, scheme):
    trial, root = source(tmp_path), tmp_path / "out"
    url = f"{scheme}://username:connection-secret@host/path"
    (trial / "result.json").write_text(json.dumps({"connection": url}))
    (trial / "trial.log").write_text(url)
    assert retain_trial_evidence(trial, root, "one")
    for name in ("result.json", "logs/trial.log"):
        retained = (root / "one/harbor" / name).read_text()
        assert "connection-secret" not in retained
        assert f"{scheme}://[REDACTED]@host/path" in retained


def test_explicit_short_credentials_are_still_secret(tmp_path):
    trial, root = source(tmp_path), tmp_path / "out"
    (trial / "trial.log").write_text("secret value: abc")
    assert retain_trial_evidence(trial, root, "one", api_key="abc")
    assert "abc" not in (root / "one/harbor/logs/trial.log").read_text()


def test_unreadable_steps_publish_an_explicit_partial_manifest(tmp_path, monkeypatch):
    trial, root = source(tmp_path), tmp_path / "out"
    steps = trial / "steps"
    steps.mkdir()
    original = Path.iterdir

    def denied(path):
        if path == steps:
            raise PermissionError("private message must not be retained")
        return original(path)

    monkeypatch.setattr(Path, "iterdir", denied)
    assert not retain_trial_evidence(trial, root, "one")
    manifest = json.loads((root / "one/harbor/manifest.json").read_text())
    assert manifest["errors"] == ["unreadable_native_steps"]
    assert not manifest["complete"]


@pytest.mark.parametrize("operation", ["verify", "retain"])
def test_parent_link_swap_cannot_escape_evidence_root(tmp_path, monkeypatch, operation):
    trial, root = source(tmp_path), tmp_path / "out"
    assert retain_trial_evidence(trial, root, "one")
    destination = root / "one/harbor"
    assert verify_trial_evidence(destination, "one")["complete"]
    directory = destination / "logs/agent" if operation == "verify" else trial / "agent"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "trajectory.json").write_text("outside-private-data")
    opened = os.open
    swapped = False

    def swap(path, flags, mode=0o777, *, dir_fd=None):
        nonlocal swapped
        if path == "agent" and dir_fd is not None and not swapped:
            directory.rename(directory.with_name("old-agent"))
            directory.symlink_to(outside, target_is_directory=True)
            swapped = True
        return opened(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swap)
    if operation == "verify":
        with pytest.raises(ValueError, match="unsafe"):
            verify_trial_evidence(destination, "one")
    else:
        assert not retain_trial_evidence(trial, root, "one")
        assert "outside-private-data" not in "".join(
            path.read_text() for path in destination.rglob("*") if path.is_file()
        )
    assert swapped


def test_fifo_manifest_is_rejected_without_blocking(tmp_path):
    import os

    directory = tmp_path / "one/harbor"
    directory.mkdir(parents=True)
    os.mkfifo(directory / "manifest.json")
    with pytest.raises(ValueError, match="regular files"):
        verify_trial_evidence(directory, "one")
    assert not trial_evidence_inventory(tmp_path, ["one"])["complete"]


@pytest.mark.parametrize(
    "kind",
    [
        "link",
        "binary",
        "missing_result",
        "invalid_result",
        "result_directory",
        "nested_result",
        "fifo",
    ],
)
def test_skipped_files_publish_explicit_incomplete_manifest(tmp_path, kind):
    import os

    trial, root = source(tmp_path), tmp_path / "out"
    if kind == "link":
        secret = tmp_path / "private"
        secret.write_text("private")
        (trial / "agent/link").symlink_to(secret)
    elif kind == "binary":
        (trial / "agent/binary").write_bytes(b"\xff\xfe")
    elif kind == "missing_result":
        (trial / "result.json").unlink()
    elif kind == "invalid_result":
        (trial / "result.json").write_text("not json")
    elif kind == "result_directory":
        (trial / "result.json").unlink()
        (trial / "result.json").mkdir()
        (trial / "result.json/data.json").write_text("{}")
    elif kind == "nested_result":
        (trial / "result.json").write_text(
            '{"x":' + "[" * 2000 + "0" + "]" * 2000 + "}"
        )
    else:
        os.mkfifo(trial / "agent/pipe")
    assert not retain_trial_evidence(trial, root, "one")
    manifest = json.loads((root / "one/harbor/manifest.json").read_text())
    assert not manifest["complete"] and manifest["errors"]
    with pytest.raises(ValueError, match="incomplete"):
        verify_trial_evidence(root / "one/harbor", "one")


@pytest.mark.parametrize(
    "change", ["missing", "extra", "modified", "traversal", "link", "duplicate"]
)
def test_download_verification_rejects_incomplete_or_unsafe_inventory(tmp_path, change):
    trial, root = source(tmp_path), tmp_path / "out"
    assert retain_trial_evidence(trial, root, "one")
    destination = root / "one/harbor"
    assert verify_trial_evidence(destination, "one")["complete"]
    result = destination / "result.json"
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if change == "missing":
        result.unlink()
    elif change == "extra":
        (destination / "unlisted.txt").write_text("extra")
    elif change == "modified":
        result.write_text("{}")
    elif change == "traversal":
        manifest["files"][0]["path"] = "../private"
        manifest_path.write_text(json.dumps(manifest))
    elif change == "duplicate":
        manifest["files"].append(manifest["files"][0])
        manifest_path.write_text(json.dumps(manifest))
    else:
        result.unlink()
        result.symlink_to(trial / "result.json")
    with pytest.raises(ValueError):
        verify_trial_evidence(destination, "one")


def test_partial_retry_replaces_inventory_and_preserves_step_evidence(tmp_path):
    trial, root = source(tmp_path), tmp_path / "out"
    (trial / "agent/binary").write_bytes(b"\xff")
    assert not retain_trial_evidence(trial, root, "one")
    (trial / "agent/binary").unlink()
    (trial / "steps/solve/agent").mkdir(parents=True)
    (trial / "steps/solve/agent/session.json").write_text('{"tokens": 5}')
    assert retain_trial_evidence(trial, root, "one")
    assert verify_trial_evidence(root / "one/harbor", "one")["complete"]
    assert not (root / "one/harbor/logs/agent/binary").exists()
    assert (root / "one/harbor/logs/steps/solve/agent/session.json").is_file()


def test_destination_links_and_rollout_traversal_cannot_write_elsewhere(tmp_path):
    trial, root = source(tmp_path), tmp_path / "out"
    root.mkdir()
    target = tmp_path / "elsewhere"
    target.mkdir()
    (root / "one").symlink_to(target)
    with pytest.raises(ValueError):
        retain_trial_evidence(trial, root, "one")
    with pytest.raises(ValueError):
        retain_trial_evidence(trial, root, "../escaped")
    assert not list(target.iterdir())


@pytest.mark.parametrize("linked_parent", [False, True])
def test_linked_artifact_root_cannot_write_elsewhere(tmp_path, linked_parent):
    trial = source(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    link = tmp_path / "linked"
    link.symlink_to(outside, target_is_directory=True)
    root = link / "artifacts" if linked_parent else link
    with pytest.raises(ValueError, match="Linked artifact root"):
        retain_trial_evidence(trial, root, "one")
    assert not list(outside.iterdir())


def test_wrong_file_size_is_rejected_before_hashing(tmp_path, monkeypatch):
    from osmosis_ai.rollout.utils import evidence

    trial, root = source(tmp_path), tmp_path / "out"
    assert retain_trial_evidence(trial, root, "one")
    destination = root / "one/harbor"
    manifest = verify_trial_evidence(destination, "one")
    first = destination / manifest["files"][0]["path"]
    with first.open("ab") as stream:
        stream.truncate(1024**4)  # Sparse hostile file: do not stream a terabyte.

    def unexpected_hash():
        pytest.fail("file size must be checked before hashing")

    monkeypatch.setattr(evidence.hashlib, "sha256", unexpected_hash)
    with pytest.raises(ValueError, match="file size mismatch"):
        verify_trial_evidence(destination, "one")


def test_declared_file_size_cannot_exceed_export_limit(tmp_path):
    from osmosis_ai.rollout.utils.evidence import MAX_EVIDENCE_FILE_BYTES

    trial, root = source(tmp_path), tmp_path / "out"
    assert retain_trial_evidence(trial, root, "one")
    destination = root / "one/harbor"
    manifest = verify_trial_evidence(destination, "one")
    manifest["files"][0]["size_bytes"] = MAX_EVIDENCE_FILE_BYTES + 1
    (destination / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="file record is invalid"):
        verify_trial_evidence(destination, "one")


def test_excessively_nested_manifest_is_incomplete_evidence(tmp_path):
    destination = tmp_path / "one/harbor"
    destination.mkdir(parents=True)
    (destination / "manifest.json").write_text("[" * 2000 + "0" + "]" * 2000)
    assert trial_evidence_inventory(tmp_path, ["one"])["missing_rollout_ids"] == ["one"]


def test_previous_process_evidence_cannot_satisfy_new_drain(tmp_path):
    trial, root = source(tmp_path), tmp_path / "out"
    assert retain_trial_evidence(trial, root, "one", process_id="old")
    assert verify_trial_evidence(root / "one/harbor", "one", process_id="old")[
        "complete"
    ]
    with pytest.raises(ValueError):
        verify_trial_evidence(root / "one/harbor", "one", process_id="new")
    inventory = trial_evidence_inventory(root, ["one"], process_id="new")
    assert not inventory["complete"] and inventory["missing_rollout_ids"] == ["one"]


@pytest.mark.parametrize("rollout_id", ["run:1", "run\n1", "run\x7f1", "run\x851"])
def test_native_identity_validation_matches_harbor_admission(tmp_path, rollout_id):
    trial, root = source(tmp_path), tmp_path / "out"
    with pytest.raises(ValueError, match="safe relative path"):
        retain_trial_evidence(trial, root, rollout_id)
    with pytest.raises(ValueError, match="safe relative path"):
        verify_trial_evidence(root, rollout_id)
    with pytest.raises(ValueError, match="safe relative path"):
        trial_evidence_inventory(root, [rollout_id])
    assert not root.exists()


def test_sanitization_limit_is_explicit_incompleteness(tmp_path, monkeypatch):
    from osmosis_ai.rollout.backend.harbor import evidence

    trial, root = source(tmp_path), tmp_path / "out"
    monkeypatch.setattr(evidence, "MAX_NATIVE_FILE_BYTES", 10)
    assert not retain_trial_evidence(trial, root, "one")
    assert not trial_evidence_inventory(root, ["one"])["complete"]


@pytest.mark.parametrize(
    "body",
    [
        [],
        None,
        {
            "schema_version": "harbor-evidence-v1",
            "rollout_id": "one",
            "complete": True,
            "errors": [],
            "files": [None],
        },
    ],
)
def test_malformed_evidence_inventory_is_rejected(tmp_path, body):
    root = tmp_path / "out"
    destination = root / "one/harbor"
    destination.mkdir(parents=True)
    (destination / "manifest.json").write_text(json.dumps(body))
    assert not trial_evidence_inventory(root, ["one"])["complete"]
