from copy import deepcopy

import pytest
from pydantic import ValidationError

from osmosis_ai.source_images import SourceImageBuildResult, SourceImageManifest


def manifest():
    image = {
        "key": "env",
        "image": "registry.example/env@sha256:" + "a" * 64,
        "context_sha256": "b" * 64,
    }
    return {
        "schema_version": "source-v1",
        "hash_policy": "harbor-v1",
        "source": {
            "repository": "https://github.com/Acme/tasks",
            "path": "tasks",
            "revision": "a" * 40,
        },
        "images": [image],
        "named_images": {"runtime": image["image"]},
        "tasks": [{"task": "group/one", "environments": {"environment": "env"}}],
    }


def test_raw_and_resolved_manifests_are_distinct_and_preserve_identity():
    raw = SourceImageManifest.model_validate(manifest())
    resolved = raw.resolved()
    assert resolved.source.repository == "https://github.com/acme/tasks"
    assert resolved.select_task_ids(["group/one", "group/one"]) == ["group/one"]
    assert (
        resolved.tasks["group/one"]["environment"].model_dump()["context_sha256"]
        == "b" * 64
    )
    with pytest.raises(ValidationError, match="valid dictionary"):
        SourceImageBuildResult.model_validate(manifest())
    with pytest.raises(ValidationError, match="valid list"):
        SourceImageManifest.model_validate(resolved.model_dump())
    with pytest.raises(ValueError, match="Select task IDs"):
        resolved.select_task_ids(["unknown"])


@pytest.mark.parametrize("field", ["key", "image"])
def test_validated_source_image_identity_cannot_be_changed(field):
    image = SourceImageManifest.model_validate(manifest()).images[0]
    with pytest.raises(ValidationError, match="Instance is frozen"):
        setattr(image, field, "mutable-substitute")


@pytest.mark.parametrize(
    "kind,reason",
    [
        ("source", "full Git commit SHA"),
        ("task", "relative to the repository root"),
        ("image", "pinned by digest"),
        ("duplicate_task", "Duplicate source task IDs"),
        ("duplicate_image", "Duplicate source image keys"),
        ("binding", "missing a required image binding"),
        ("named", "absent from the image inventory"),
    ],
)
def test_invalid_published_inventory_is_rejected(kind, reason):
    data = manifest()
    if kind == "source":
        data["source"]["revision"] = "main"
    elif kind == "task":
        data["tasks"][0]["task"] = "../escape"
    elif kind == "image":
        data["images"][0]["image"] = "registry.example/env:latest"
    elif kind == "duplicate_task":
        data["tasks"].append(deepcopy(data["tasks"][0]))
    elif kind == "duplicate_image":
        data["images"].append(deepcopy(data["images"][0]))
    elif kind == "binding":
        data["tasks"][0]["environments"]["environment"] = "missing"
    else:
        data["named_images"]["runtime"] = "registry.example/missing@sha256:" + "c" * 64
    with pytest.raises(ValidationError, match=reason):
        SourceImageManifest.model_validate(data)


@pytest.mark.parametrize("task", [".", "", "/absolute", "x//y", "x\\y", "x\ny"])
def test_resolved_inventory_rejects_nonportable_task_ids(task):
    data = SourceImageManifest.model_validate(manifest()).resolved().model_dump()
    data["tasks"][task] = data["tasks"].pop("group/one")
    with pytest.raises(
        ValidationError, match=r"source root|relative to the repository root"
    ):
        SourceImageBuildResult.model_validate(data)


def test_resolved_binding_cannot_substitute_a_different_image():
    data = SourceImageManifest.model_validate(manifest()).resolved().model_dump()
    data["tasks"]["group/one"]["environment"]["image"] = (
        "registry.example/env@sha256:" + "f" * 64
    )
    with pytest.raises(ValidationError, match="differs from its inventory"):
        SourceImageBuildResult.model_validate(data)
