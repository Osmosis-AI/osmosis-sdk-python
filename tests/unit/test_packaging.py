"""Packaging: wheel build, caching, and bundle inspection."""

import zipfile
from pathlib import Path

import pytest
from packaging.utils import InvalidName

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
    assert "from my_harness.solver import MyWorkflow as _osmosis_workflow" in shim
    assert "runner.agent_main(_osmosis_workflow, None)" in shim

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
    cached = build_bundle(project, **kwargs)
    assert cached == first
    assert cached.stat().st_mtime_ns == first_mtime

    (project / "my_harness" / "solver.py").write_text(
        "class MyWorkflow:\n    changed = True\n"
    )
    changed = build_bundle(project, **kwargs)
    assert changed != first
    assert changed.stat().st_mtime_ns != first_mtime


def test_cache_keeps_project_versions_separate(project, tmp_path):
    bundles_dir = tmp_path / "bundles"
    kwargs = dict(workflow="my_harness.solver:MyWorkflow", bundles_dir=bundles_dir)

    first = build_bundle(project, **kwargs)
    assert first.name.startswith("my_harness-0.1.0-")

    (project / "pyproject.toml").write_text(PYPROJECT.replace("0.1.0", "0.2.0"))
    upgraded = build_bundle(project, **kwargs)

    assert upgraded != first
    assert upgraded.name.startswith("my_harness-0.2.0-")
    assert build_bundle(project, **kwargs) == upgraded


def test_invalid_cache_entry_has_recovery_instructions(project, tmp_path):
    bundles_dir = tmp_path / "bundles"
    kwargs = dict(workflow="my_harness.solver:MyWorkflow", bundles_dir=bundles_dir)
    wheel = build_bundle(project, **kwargs)
    wheel.unlink()

    with pytest.raises(RuntimeError, match="delete this directory and retry"):
        build_bundle(project, **kwargs)


def test_concurrent_builds_publish_one_cache_entry(project, tmp_path, monkeypatch):
    import threading
    from concurrent.futures import ThreadPoolExecutor

    import osmosis_ai.packaging as packaging

    hash_barrier = threading.Barrier(2)
    build_barrier = threading.Barrier(2)
    original_content_hash = packaging.content_hash

    def synchronized_content_hash(*args, **kwargs):
        value = original_content_hash(*args, **kwargs)
        hash_barrier.wait(timeout=5)
        return value

    def fake_uv_build(args, **_kwargs):
        output = Path(args[args.index("--out-dir") + 1])
        output.mkdir(parents=True)
        (output / "my_harness-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
        build_barrier.wait(timeout=5)

    monkeypatch.setattr(packaging, "content_hash", synchronized_content_hash)
    monkeypatch.setattr(packaging.subprocess, "run", fake_uv_build)
    bundles_dir = tmp_path / "bundles"

    def build() -> Path:
        return build_bundle(
            project,
            workflow="my_harness.solver:MyWorkflow",
            bundles_dir=bundles_dir,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: build(), range(2)))

    assert results[0] == results[1]
    assert results[0].is_file()


def test_cache_key_includes_explicit_package(tmp_path):
    code_dir = tmp_path / "project"
    for package in ("first", "second", "shared"):
        package_dir = code_dir / package
        package_dir.mkdir(parents=True)
        (package_dir / "__init__.py").touch()
    (code_dir / "shared" / "solver.py").write_text("class Workflow: pass\n")
    (code_dir / "pyproject.toml").write_text(
        PYPROJECT.replace('include = ["my_harness*"]', 'include = ["*"]')
    )
    bundles_dir = tmp_path / "bundles"

    first = build_bundle(
        code_dir,
        package="first",
        workflow="shared.solver:Workflow",
        bundles_dir=bundles_dir,
    )
    second = build_bundle(
        code_dir,
        package="second",
        workflow="shared.solver:Workflow",
        bundles_dir=bundles_dir,
    )

    assert first != second
    assert inspect_bundle(first).agent_script == "first-agent"
    assert inspect_bundle(second).agent_script == "second-agent"


def test_source_hash_ignores_excluded_ancestor_names(project, tmp_path):
    nested_root = tmp_path / "build"
    nested_root.mkdir()
    nested_project = project.rename(nested_root / "project")
    bundles_dir = tmp_path / "bundles"
    kwargs = dict(workflow="my_harness.solver:MyWorkflow", bundles_dir=bundles_dir)

    first = build_bundle(nested_project, **kwargs)
    (nested_project / "my_harness" / "solver.py").write_text(
        "class MyWorkflow:\n    changed = True\n"
    )
    changed = build_bundle(nested_project, **kwargs)

    assert changed != first


def test_distribution_name_uses_wheel_normalization(project, tmp_path):
    (project / "pyproject.toml").write_text(
        PYPROJECT.replace('name = "my-harness"', 'name = "Acme.Rollout"')
    )

    wheel = build_bundle(
        project,
        workflow="my_harness.solver:MyWorkflow",
        bundles_dir=tmp_path / "bundles",
    )

    assert wheel.name.startswith("acme_rollout-0.1.0-")


def test_rejects_invalid_distribution_name(project, tmp_path):
    (project / "pyproject.toml").write_text(
        PYPROJECT.replace('name = "my-harness"', 'name = "not valid!"')
    )

    with pytest.raises(InvalidName):
        build_bundle(
            project,
            workflow="my_harness.solver:MyWorkflow",
            bundles_dir=tmp_path / "bundles",
        )


def test_uv_executable_uses_active_scripts_scheme(tmp_path, monkeypatch):
    import osmosis_ai.packaging as packaging

    scripts_dir = tmp_path / "Scripts"
    scripts_dir.mkdir()
    uv = scripts_dir / "uv.exe"
    uv.touch()
    uv.chmod(0o755)

    def get_path(name):
        assert name == "scripts"
        return str(scripts_dir)

    monkeypatch.setattr(packaging.sysconfig, "get_path", get_path)
    monkeypatch.setattr(packaging.sys, "executable", str(tmp_path / "python.exe"))
    monkeypatch.setattr(packaging.shutil, "which", lambda _name: None)

    assert packaging._uv_executable() == str(uv)


def test_uv_executable_reports_missing_builder(tmp_path, monkeypatch):
    import osmosis_ai.packaging as packaging

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    monkeypatch.setattr(packaging.sysconfig, "get_path", lambda name: str(scripts_dir))
    monkeypatch.setattr(packaging.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(packaging.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match=r"install osmosis-ai\[harbor\]"):
        packaging._uv_executable()


def test_uv_executable_skips_non_executable_candidate(tmp_path, monkeypatch):
    import osmosis_ai.packaging as packaging

    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    candidate = scripts_dir / "uv"
    candidate.touch()
    fallback = tmp_path / "path" / "uv"
    monkeypatch.setattr(packaging.sysconfig, "get_path", lambda name: str(scripts_dir))
    monkeypatch.setattr(packaging.sys, "executable", str(tmp_path / "python"))
    monkeypatch.setattr(packaging.os, "access", lambda path, mode: path != candidate)
    monkeypatch.setattr(packaging.shutil, "which", lambda _name: str(fallback))

    assert packaging._uv_executable() == str(fallback)


@pytest.mark.parametrize("wheel_names", [[], ["one.whl", "two.whl"]])
def test_build_requires_exactly_one_wheel(project, tmp_path, monkeypatch, wheel_names):
    import osmosis_ai.packaging as packaging

    def fake_uv_build(args, **_kwargs):
        output = Path(args[args.index("--out-dir") + 1])
        output.mkdir(parents=True)
        for name in wheel_names:
            (output / name).touch()

    monkeypatch.setattr(packaging, "_uv_executable", lambda: "uv")
    monkeypatch.setattr(packaging.subprocess, "run", fake_uv_build)

    with pytest.raises(RuntimeError, match=rf"produced {len(wheel_names)} wheels"):
        build_bundle(
            project,
            workflow="my_harness.solver:MyWorkflow",
            bundles_dir=tmp_path / "bundles",
        )


def test_generated_shim_aliases_same_named_references(project, tmp_path):
    (project / "my_harness" / "solver.py").write_text("class Entry: pass\n")
    (project / "my_harness" / "grade.py").write_text("class Entry: pass\n")
    (project / "my_harness" / "workflow_config.py").write_text("Entry = object()\n")
    (project / "my_harness" / "grader_config.py").write_text("Entry = object()\n")

    wheel = build_bundle(
        project,
        workflow="my_harness.solver:Entry",
        grader="my_harness.grade:Entry",
        workflow_config="my_harness.workflow_config:Entry",
        grader_config="my_harness.grader_config:Entry",
        bundles_dir=tmp_path / "bundles",
    )
    with zipfile.ZipFile(wheel) as archive:
        shim = archive.read("my_harness/bundle_main.py").decode()

    assert "from my_harness.solver import Entry as _osmosis_workflow" in shim
    assert "from my_harness.grade import Entry as _osmosis_grader" in shim
    assert (
        "from my_harness.workflow_config import Entry as _osmosis_workflow_config"
        in shim
    )
    assert (
        "from my_harness.grader_config import Entry as _osmosis_grader_config" in shim
    )
    assert "runner.agent_main(_osmosis_workflow, _osmosis_workflow_config)" in shim
    assert "runner.grader_main(_osmosis_grader, _osmosis_grader_config)" in shim


def test_rejects_bad_refs_and_missing_pyproject(project, tmp_path):
    with pytest.raises(ValueError, match="module:attr"):
        build_bundle(project, workflow="not-a-ref", bundles_dir=tmp_path / "b")
    with pytest.raises(ValueError, match=r"pyproject\.toml"):
        build_bundle(tmp_path / "nowhere", workflow="a:B", bundles_dir=tmp_path / "b")


SRC_PYPROJECT = PYPROJECT.replace(
    "[tool.setuptools.packages.find]\ninclude",
    '[tool.setuptools.packages.find]\nwhere = ["src"]\ninclude',
)


@pytest.fixture
def src_project(tmp_path):
    """A ``src/`` layout project — what ``uv init --lib`` scaffolds."""
    code_dir = tmp_path / "harness"
    package = code_dir / "src" / "my_harness"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    (package / "solver.py").write_text("class MyWorkflow: pass\n")
    (package / "grade.py").write_text("class MyGrader: pass\n")
    (code_dir / "pyproject.toml").write_text(SRC_PYPROJECT)
    return code_dir


def test_bundles_dir_inside_project_does_not_recurse(project, tmp_path):
    # The staging tree lives under bundles_dir; if that sits inside the project
    # then copytree's destination is a descendant of its own source, and the
    # published wheels feed back into the cache key.
    bundles_dir = project / "bundles"
    kwargs = dict(workflow="my_harness.solver:MyWorkflow", bundles_dir=bundles_dir)

    first = build_bundle(project, **kwargs)

    with zipfile.ZipFile(first) as archive:
        names = archive.namelist()
    # The staging tree is torn down before build_bundle returns, so the only
    # durable evidence of a recursive copy is what landed in the wheel.
    assert "my_harness/solver.py" in names
    assert not [n for n in names if "bundles/" in n]
    # The cache must still hit: excluding it from copytree alone is not enough,
    # since content_hash() would otherwise digest the wheel it just published.
    assert build_bundle(project, **kwargs) == first


def test_bundles_dir_inside_project_still_tracks_source_changes(project, tmp_path):
    bundles_dir = project / "bundles"
    kwargs = dict(workflow="my_harness.solver:MyWorkflow", bundles_dir=bundles_dir)

    first = build_bundle(project, **kwargs)
    (project / "my_harness" / "solver.py").write_text(
        "class MyWorkflow:\n    changed = True\n"
    )

    assert build_bundle(project, **kwargs) != first


def test_src_layout_package_is_detected_and_shimmed(src_project, tmp_path):
    wheel = build_bundle(
        src_project,
        workflow="my_harness.solver:MyWorkflow",
        grader="my_harness.grade:MyGrader",
        bundles_dir=tmp_path / "bundles",
    )

    with zipfile.ZipFile(wheel) as archive:
        # The wheel is import-rooted, so the shim lands at my_harness/ even
        # though it was written into src/my_harness/ in the staging tree.
        shim = archive.read("my_harness/bundle_main.py").decode()
    assert "from my_harness.solver import MyWorkflow as _osmosis_workflow" in shim
    assert "from my_harness.grade import MyGrader as _osmosis_grader" in shim

    info = inspect_bundle(wheel)
    assert info.agent_script == "my-harness-agent"
    assert info.grader_script == "my-harness-grade"


def test_namespace_package_keeps_working_without_init(tmp_path):
    # PEP 420 packages have no __init__.py; auto-detection has never found
    # them, but an explicit package= used to work and must keep working.
    from osmosis_ai.packaging import find_package_dir

    (tmp_path / "ns_harness").mkdir()

    assert find_package_dir(tmp_path, "ns_harness") == tmp_path / "ns_harness"


def test_src_layout_accepts_explicit_package(src_project, tmp_path):
    wheel = build_bundle(
        src_project,
        workflow="my_harness.solver:MyWorkflow",
        package="my_harness",
        bundles_dir=tmp_path / "bundles",
    )

    with zipfile.ZipFile(wheel) as archive:
        assert "my_harness/bundle_main.py" in archive.namelist()


def test_find_package_error_names_every_searched_root(tmp_path):
    from osmosis_ai.packaging import find_package

    code_dir = tmp_path / "empty"
    (code_dir / "src").mkdir(parents=True)

    with pytest.raises(ValueError, match="found none") as exc:
        find_package(code_dir)
    assert str(code_dir / "src") in str(exc.value)


def test_find_package_dir_rejects_unknown_package(tmp_path):
    from osmosis_ai.packaging import find_package_dir

    (tmp_path / "src").mkdir()

    with pytest.raises(ValueError, match="'nope' not found"):
        find_package_dir(tmp_path, "nope")


def test_project_dir_for_walks_out_of_src_layout(tmp_path, monkeypatch):
    import sys

    from osmosis_ai.packaging import project_dir_for

    code_dir = tmp_path / "harness"
    package = code_dir / "src" / "my_harness"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    (package / "solver.py").write_text("class MyWorkflow: pass\n")
    (code_dir / "pyproject.toml").write_text(SRC_PYPROJECT)

    monkeypatch.syspath_prepend(str(code_dir / "src"))
    sys.modules.pop("my_harness", None)
    sys.modules.pop("my_harness.solver", None)
    try:
        from my_harness.solver import MyWorkflow  # type: ignore[import-not-found]

        assert project_dir_for(MyWorkflow) == code_dir
    finally:
        sys.modules.pop("my_harness", None)
        sys.modules.pop("my_harness.solver", None)


def test_project_dir_for_locates_bench_harness():
    import sys
    from pathlib import Path

    from osmosis_ai.packaging import project_dir_for

    harness_dir = (
        Path(__file__).parents[2]
        / "benchmarks"
        / "container_lifecycle"
        / "bench_harness"
    )
    sys.path.insert(0, str(harness_dir))
    try:
        from bench_harness.solver import BenchWorkflow

        assert project_dir_for(BenchWorkflow) == harness_dir
    finally:
        sys.path.remove(str(harness_dir))
