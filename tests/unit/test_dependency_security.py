"""Published dependency floors and feature ownership for security fixes."""

from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

REPO_ROOT = Path(__file__).parents[2]
FSSPEC_EXTRAS = {"strands", "openai-agents", "harbor", "rubric", "eval"}


def test_fsspec_security_floor_belongs_to_each_owning_extra() -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    assert all(Requirement(raw).name != "fsspec" for raw in project["dependencies"]), (
        "The fsspec security floor must not expand the base installation"
    )

    owning_extras = set()
    for extra, requirements in project["optional-dependencies"].items():
        for raw in requirements:
            requirement = Requirement(raw)
            if requirement.name != "fsspec":
                continue
            owning_extras.add(extra)
            assert requirement.marker is None, "The floor must cover every platform"
            assert ">=2026.6.0" in str(requirement.specifier).split(",")
            assert not requirement.specifier.contains("2026.1.0")
            assert not requirement.specifier.contains("2026.5.0")
            assert requirement.specifier.contains("2026.6.0")
    assert owning_extras == FSSPEC_EXTRAS


def test_locked_fsspec_meets_security_floor() -> None:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    packages = [package for package in lock["package"] if package["name"] == "fsspec"]
    assert packages, "The owning extras must still resolve fsspec"
    assert all(
        Version(package["version"]) >= Version("2026.6.0") for package in packages
    )
