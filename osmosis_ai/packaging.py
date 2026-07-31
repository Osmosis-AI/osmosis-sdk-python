"""Build an installable wheel from an agent/harness project dir.

Stages the user's project (their pyproject.toml, dependencies, and build
backend included), injects a generated shim with literal imports, and exposes
``<package>-agent`` / ``<package>-grade`` console scripts. The wheel installs
anywhere with one ``pip install`` — a rollout container or a user's own box.
"""

from __future__ import annotations

import configparser
import hashlib
import shutil
import subprocess
import tempfile
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import toml

BUNDLES_DIR = Path.home() / ".osmosis" / "bundles"

SHIM_TEMPLATE = """\
{imports}
from osmosis_ai.rollout.container import runner


def agent_main():
    runner.agent_main({workflow_class}, {workflow_config})


def grader_main():
    runner.grader_main({grader_class}, {grader_config})
"""

EXCLUDE_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "dist",
    "build",
    "node_modules",
    ".pytest_cache",
    ".ruff_cache",
}


@dataclass(frozen=True)
class BundleInfo:
    wheel: Path
    agent_script: str
    grader_script: str | None


def agent_script_name(package: str) -> str:
    return f"{package.replace('_', '-')}-agent"


def grader_script_name(package: str) -> str:
    return f"{package.replace('_', '-')}-grade"


def content_hash(path: Path, *, extra: str = "") -> str:
    digest = hashlib.sha256(extra.encode())
    for file in sorted(
        p for p in path.rglob("*") if p.is_file() and not (set(p.parts) & EXCLUDE_DIRS)
    ):
        digest.update(str(file.relative_to(path)).encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()[:32]


def find_package(code_dir: Path) -> str:
    packages = sorted(
        d.name
        for d in code_dir.iterdir()
        if d.is_dir() and (d / "__init__.py").exists() and d.name not in EXCLUDE_DIRS
    )
    if len(packages) != 1:
        raise ValueError(
            f"expected exactly one python package in {code_dir}, "
            f"found {packages or 'none'}; pass package= explicitly"
        )
    return packages[0]


def split_ref(ref: str, label: str) -> tuple[str, str]:
    if ":" not in ref:
        raise ValueError(f"{label} must be 'module:attr', got {ref!r}")
    module, attr = ref.rsplit(":", 1)
    return module, attr


def project_dir_for(obj: object) -> Path:
    """Locate the project dir (containing pyproject.toml) of *obj*'s package."""
    import inspect

    package_dir = Path(inspect.getfile(type(obj) if not isinstance(obj, type) else obj)).parent
    while (package_dir.parent / "__init__.py").exists():
        package_dir = package_dir.parent
    project_dir = package_dir.parent
    if not (project_dir / "pyproject.toml").is_file():
        raise ValueError(
            f"no pyproject.toml next to package {package_dir.name!r} "
            f"(looked in {project_dir}); pass code_dir= explicitly"
        )
    return project_dir


def inspect_bundle(wheel: Path) -> BundleInfo:
    """Read the bundle's console script names from its wheel metadata."""
    with zipfile.ZipFile(wheel) as archive:
        entry = next(
            name for name in archive.namelist() if name.endswith("entry_points.txt")
        )
        parser = configparser.ConfigParser()
        parser.read_string(archive.read(entry).decode())
    scripts = dict(parser.items("console_scripts"))
    agent = next(
        (n for n, t in scripts.items() if t.endswith(".bundle_main:agent_main")), None
    )
    grader = next(
        (n for n, t in scripts.items() if t.endswith(".bundle_main:grader_main")), None
    )
    if agent is None:
        raise ValueError(f"{wheel.name} is not an osmosis bundle (no agent script)")
    return BundleInfo(wheel=wheel, agent_script=agent, grader_script=grader)


def build_bundle(
    code_dir: Path,
    *,
    workflow: str,
    grader: str | None = None,
    workflow_config: str | None = None,
    grader_config: str | None = None,
    deps: list[str] | None = None,
    package: str | None = None,
    bundles_dir: Path = BUNDLES_DIR,
) -> Path:
    """Package the project at *code_dir* into a wheel; rebuild only on change.

    workflow/grader are 'module:Class' refs; the config refs point at
    module-level instances and are passed to the runner as-is.
    """
    code_dir = code_dir.resolve()
    pyproject_path = code_dir / "pyproject.toml"
    if not pyproject_path.is_file():
        raise ValueError(f"project dir must contain pyproject.toml: {code_dir}")
    package = package or find_package(code_dir)

    imports = []
    references = {}
    for label, ref in (
        ("workflow", workflow),
        ("grader", grader),
        ("workflow_config", workflow_config),
        ("grader_config", grader_config),
    ):
        if ref is None:
            references[label] = "None"
            continue
        module, attr = split_ref(ref, label)
        imports.append(f"from {module} import {attr}")
        references[label] = attr
    shim = SHIM_TEMPLATE.format(
        imports="\n".join(imports),
        workflow_class=references["workflow"],
        workflow_config=references["workflow_config"],
        grader_class=references["grader"],
        grader_config=references["grader_config"],
    )

    scripts = {agent_script_name(package): f"{package}.bundle_main:agent_main"}
    if grader:
        scripts[grader_script_name(package)] = f"{package}.bundle_main:grader_main"

    project = tomllib.loads(pyproject_path.read_text()).get("project", {})
    name = project.get("name") or f"osmosis-bundle-{package}"
    bundle_key = content_hash(
        code_dir, extra=pyproject_path.read_text() + shim + repr(deps or [])
    )
    bundles_dir.mkdir(parents=True, exist_ok=True)
    wheel_glob = f"{name.replace('-', '_')}-*.whl"
    marker = bundles_dir / f"{name}-{bundle_key}.key"
    cached = sorted(bundles_dir.glob(wheel_glob))
    if cached and marker.exists():
        return cached[0]

    for old in bundles_dir.glob(f"{name}-*.key"):
        old.unlink()
    with tempfile.TemporaryDirectory() as staging:
        stage = Path(staging)
        shutil.copytree(
            code_dir,
            stage,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns(*EXCLUDE_DIRS, "*.pyc"),
        )
        (stage / package / "bundle_main.py").write_text(shim)
        staged = tomllib.loads((stage / "pyproject.toml").read_text())
        staged_project = staged.setdefault("project", {})
        staged_project["dependencies"] = [
            *staged_project.get("dependencies", []),
            *(deps or []),
        ]
        staged_project["scripts"] = {**staged_project.get("scripts", {}), **scripts}
        (stage / "pyproject.toml").write_text(toml.dumps(staged))
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(bundles_dir), str(stage)],
            check=True,
            capture_output=True,
        )

    marker.touch()
    wheels = sorted(bundles_dir.glob(wheel_glob))
    if not wheels:
        raise RuntimeError(f"uv build produced no wheel in {bundles_dir}")
    return wheels[0]
