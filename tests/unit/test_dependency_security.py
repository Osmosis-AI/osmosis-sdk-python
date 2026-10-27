"""Published dependency floors and feature ownership for security fixes."""

from __future__ import annotations

import runpy
import tomllib
from importlib.metadata import Distribution, distribution
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
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


@pytest.mark.parametrize(
    ("name", "extra", "floor"),
    [
        ("certifi", None, "2026.10.19"),
        ("uv", "harbor", "0.13.4"),
        ("uv", "eval", "0.13.4"),
    ],
)
def test_ca_bundle_floors_allow_patched_releases(
    name: str, extra: str | None, floor: str
) -> None:
    project = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())["project"]
    requirements = (
        project["dependencies"]
        if extra is None
        else project["optional-dependencies"][extra]
    )
    matches = [
        Requirement(raw) for raw in requirements if Requirement(raw).name == name
    ]
    assert len(matches) == 1
    requirement = matches[0]
    assert requirement.marker is None, "The security floor must cover every platform"
    assert f">={floor}" in str(requirement.specifier).split(",")
    assert floor in requirement.specifier, (
        "The upper bound must admit the patched release"
    )


@pytest.mark.parametrize(
    ("name", "constraint"),
    [
        ("certifi", ">=2026.10.19"),
        ("uv", ">=0.13.4"),
        ("filelock", ">=4.2.1"),
        ("litellm", ">=1.103.1,!=1.104.0rc1"),
        ("opentelemetry-api", ">=1.45.1"),
        ("strands-agents", ">=1.48.0"),
        ("virtualenv", ">=21.14.13"),
    ],
)
def test_locked_dependencies_meet_security_constraints(
    name: str, constraint: str
) -> None:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text())
    packages = [package for package in lock["package"] if package["name"] == name]
    assert packages, f"The lock must still resolve {name}"
    assert all(
        Version(package["version"]) in SpecifierSet(constraint) for package in packages
    )


@pytest.mark.parametrize(
    ("package", "constraint", "safe"),
    [
        ("aiohttp", ">=3.14.5", True),
        ("aiohttp", "~=3.14.4", True),
        ("certifi", ">=2026.10.19", True),
        ("certifi", ">=2026.10.20", True),
        ("multidict", ">=6.10.0", True),
        ("mcp", ">=1.31.0,<2", True),
        ("mcp", ">=2.2.0", True),
        ("uv", ">=0.13.4,<0.14", True),
        ("uv", "~=0.13.5", True),
        ("filelock", ">=4.2.1", True),
        ("litellm", ">=1.103.1,!=1.104.0rc1,<2", True),
        ("opentelemetry-api", ">=1.45.1,<2", True),
        ("strands-agents", ">=1.48.0,<2", True),
        ("aiohttp", ">=3.14.3", False),
        ("certifi", ">=2025.11.12", False),
        ("certifi", ">=2026.10.18", False),
        ("multidict", ">=6.7.0", False),
        ("multidict", ">=6.9.0", False),
        ("mcp", ">=1.29.1,<2", False),
        ("mcp", ">=1.30.0", False),
        ("mcp", ">=1.30.0,!=2.0.*", False),
        ("mcp", ">=1.30.0,!=2.1.*", False),
        ("uv", ">=0.9.7,<0.13", False),
        ("uv", ">=0.13.3,<0.14", False),
        ("filelock", ">=4.2.0", False),
        ("litellm", ">=1.100.4,<2", False),
        ("litellm", ">=1.103.1,<2", False),
        ("opentelemetry-api", ">=1.45.0,<2", False),
        ("strands-agents", ">=1.43.0,<2", False),
    ],
)
def test_wheel_verifier_checks_allowed_versions(
    tmp_path: Path, package: str, constraint: str, safe: bool
) -> None:
    metadata = distribution("osmosis-ai").metadata
    requirements = metadata.get_all("Requires-Dist", [])
    del metadata["Requires-Dist"]
    for raw in requirements:
        requirement = Requirement(raw)
        if requirement.name == package:
            requirement.specifier = SpecifierSet(constraint)
        metadata["Requires-Dist"] = str(requirement)
    (tmp_path / "METADATA").write_text(metadata.as_string())

    verifier = runpy.run_path(
        str(REPO_ROOT / ".github/scripts/verify-wheel-install.py")
    )["_assert_dependency_metadata"]
    if safe:
        verifier(Distribution.at(tmp_path))
    else:
        with pytest.raises(
            AssertionError, match=f"{package} requirement allows vulnerable"
        ):
            verifier(Distribution.at(tmp_path))
