"""Packaging: wheel build, caching, and bundle inspection."""

import zipfile

import pytest

from osmosis_ai.packaging import build_bundle, inspect_bundle

PYPROJECT = """\
[project]
name = "my-harness"
version = "0.1.0"
dependencies = ["httpx>=0.27"]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["my_harness*"]
"""


@pytest.fixture
def project(tmp_path):
    code_dir = tmp_path / "harness"
    package = code_dir / "my_harness"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    (package / "solver.py").write_text("class MyWorkflow: pass\n")
    (package / "grade.py").write_text("class MyGrader: pass\n")
    (code_dir / "pyproject.toml").write_text(PYPROJECT)
    return code_dir


def test_build_produces_scripts_and_keeps_deps(project, tmp_path):
    wheel = build_bundle(
        project,
        workflow="my_harness.solver:MyWorkflow",
        grader="my_harness.grade:MyGrader",
        bundles_dir=tmp_path / "bundles",
    )
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata = archive.read(
            next(n for n in names if n.endswith("METADATA"))
        ).decode()
        shim = archive.read("my_harness/bundle_main.py").decode()

    assert "my_harness/solver.py" in names
    assert "Requires-Dist: httpx>=0.27" in metadata
    assert "from my_harness.solver import MyWorkflow" in shim
    assert "runner.agent_main(MyWorkflow, None)" in shim

    info = inspect_bundle(wheel)
    assert info.agent_script == "my-harness-agent"
    assert info.grader_script == "my-harness-grade"


def test_deps_override_same_named_pyproject_entries(project, tmp_path):
    wheel = build_bundle(
        project,
        workflow="my_harness.solver:MyWorkflow",
        deps=["httpx @ https://example.com/httpx.tar.gz"],
        bundles_dir=tmp_path / "bundles",
    )
    with zipfile.ZipFile(wheel) as archive:
        metadata = archive.read(
            next(n for n in archive.namelist() if n.endswith("METADATA"))
        ).decode()
    assert "Requires-Dist: httpx>=0.27" not in metadata
    assert "Requires-Dist: httpx @ https://example.com/httpx.tar.gz" in metadata


def test_requirements_keep_markers_drop_extras(tmp_path):
    code_dir = tmp_path / "harness"
    package = code_dir / "my_harness"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    (package / "solver.py").write_text("class W: pass\n")
    (code_dir / "pyproject.toml").write_text(
        """\
[project]
name = "my-harness"
version = "0.1.0"
dependencies = ["httpx>=0.27", "tomli; python_version < '3.11'"]

[project.optional-dependencies]
dev = ["pytest"]

[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
include = ["my_harness*"]
"""
    )
    wheel = build_bundle(
        code_dir, workflow="my_harness.solver:W", bundles_dir=tmp_path / "bundles"
    )
    requirements = inspect_bundle(wheel).requirements
    assert "httpx>=0.27" in requirements
    assert any(r.startswith("tomli") for r in requirements)
    assert not any("pytest" in r for r in requirements)


def test_grader_optional(project, tmp_path):
    wheel = build_bundle(
        project,
        workflow="my_harness.solver:MyWorkflow",
        bundles_dir=tmp_path / "bundles",
    )
    info = inspect_bundle(wheel)
    assert info.grader_script is None


def test_cache_hits_until_source_changes(project, tmp_path):
    bundles_dir = tmp_path / "bundles"
    kwargs = dict(workflow="my_harness.solver:MyWorkflow", bundles_dir=bundles_dir)

    first = build_bundle(project, **kwargs)
    first_mtime = first.stat().st_mtime_ns
    assert build_bundle(project, **kwargs).stat().st_mtime_ns == first_mtime

    (project / "my_harness" / "solver.py").write_text(
        "class MyWorkflow:\n    changed = True\n"
    )
    assert build_bundle(project, **kwargs).stat().st_mtime_ns != first_mtime


def test_rejects_bad_refs_and_missing_pyproject(project, tmp_path):
    with pytest.raises(ValueError, match="module:attr"):
        build_bundle(project, workflow="not-a-ref", bundles_dir=tmp_path / "b")
    with pytest.raises(ValueError, match=r"pyproject\.toml"):
        build_bundle(tmp_path / "nowhere", workflow="a:B", bundles_dir=tmp_path / "b")


def test_project_dir_for_locates_bench_harness():
    import sys
    from pathlib import Path

    from osmosis_ai.packaging import project_dir_for

    harness_dir = (
        Path(__file__).parents[2] / "benchmarks" / "container_lifecycle" / "bench_harness"
    )
    sys.path.insert(0, str(harness_dir))
    try:
        from bench_harness.solver import BenchWorkflow

        assert project_dir_for(BenchWorkflow) == harness_dir
    finally:
        sys.path.remove(str(harness_dir))
