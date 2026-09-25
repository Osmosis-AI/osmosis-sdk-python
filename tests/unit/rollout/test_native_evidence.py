from __future__ import annotations

import json

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


@pytest.mark.parametrize(
    "kind", ["link", "binary", "missing_result", "invalid_result", "fifo"]
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
    retain_trial_evidence(trial, root, "one")
    destination = root / "one/harbor"
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
