"""SDK-owned workspace template catalog.

The public workspace-template repository contains user-editable starter files.
Control metadata such as recipe ownership and official scaffold contents lives
in the SDK so local user edits cannot change CLI behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent


@dataclass(frozen=True, slots=True)
class TemplateRecipe:
    """A template recipe known by this SDK version."""

    name: str
    description: str
    files: tuple[Path, ...]
    owned_dirs: tuple[Path, ...]
    next_steps: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class OfficialScaffoldFile:
    """An official agent/scaffold file managed by the SDK."""

    path: Path
    content: str


@dataclass(frozen=True, slots=True)
class ScaffoldEntry:
    """A file or directory marker needed for project repair."""

    dest: str
    content: str = ""
    official: bool = False


def _path(value: str) -> Path:
    return Path(value)


TEMPLATE_RECIPES: tuple[TemplateRecipe, ...] = (
    TemplateRecipe(
        name="multiply-local-strands",
        description="Local Strands multiply rollout",
        files=(
            _path("rollouts/multiply-local-strands/**"),
            _path("configs/eval/multiply-local-strands.toml"),
            _path("configs/training/multiply-local-strands.toml"),
            _path("data/multiply.jsonl"),
        ),
        owned_dirs=(_path("rollouts/multiply-local-strands"),),
        next_steps=(
            "pip install -e rollouts/multiply-local-strands",
            "osmosis eval run configs/eval/multiply-local-strands.toml --limit 1",
            "git push",
            "Confirm Git Sync is connected in the Osmosis Platform",
            "osmosis train submit configs/training/multiply-local-strands.toml",
        ),
    ),
    TemplateRecipe(
        name="multiply-local-openai",
        description="Local OpenAI Agents multiply rollout",
        files=(
            _path("rollouts/multiply-local-openai/**"),
            _path("configs/eval/multiply-local-openai.toml"),
            _path("configs/training/multiply-local-openai.toml"),
            _path("data/multiply.jsonl"),
        ),
        owned_dirs=(_path("rollouts/multiply-local-openai"),),
        next_steps=(
            "pip install -e rollouts/multiply-local-openai",
            "osmosis eval run configs/eval/multiply-local-openai.toml --limit 1",
            "git push",
            "Confirm Git Sync is connected in the Osmosis Platform",
            "osmosis train submit configs/training/multiply-local-openai.toml",
        ),
    ),
    TemplateRecipe(
        name="multiply-harbor-strands",
        description="Harbor-backed Strands multiply rollout",
        files=(
            _path("rollouts/multiply-harbor-strands/**"),
            _path("configs/eval/multiply-harbor-strands.toml"),
            _path("configs/training/multiply-harbor-strands.toml"),
            _path("data/multiply.jsonl"),
        ),
        owned_dirs=(_path("rollouts/multiply-harbor-strands"),),
        next_steps=(
            "pip install -e rollouts/multiply-harbor-strands",
            "osmosis eval run configs/eval/multiply-harbor-strands.toml --limit 1",
            "git push",
            "Confirm Git Sync is connected in the Osmosis Platform",
            "osmosis train submit configs/training/multiply-harbor-strands.toml",
        ),
    ),
)


AGENTS_MD = dedent(
    """\
    # Osmosis Project

    This is a **structured Osmosis project**. Do not invent a different
    top-level layout.

    ## Project contract

    - Required paths:
      - `.osmosis/project.toml`
      - `rollouts/`
      - `configs/training/`
      - `configs/eval/`
      - `data/`
    - New rollouts live in `rollouts/<name>/`.
    - The canonical rollout entrypoint is `rollouts/<name>/main.py`.
    - Eval configs live in `configs/eval/<name>.toml`.
    - Training configs live in `configs/training/<name>.toml`.
    - Local training guidance lives in `.osmosis/research/program.md`.
    - Local cache state lives in `.osmosis/cache/` and should not be treated as source.
    - Do not create new top-level directories unless the user explicitly asks.

    ## Rollout contract

    - Each rollout entrypoint must expose exactly one concrete `AgentWorkflow`.
    - Local eval and managed training require a concrete `Grader` in the rollout
      server. Eval configs do not support `[grader]` overrides.
    - Tools should be async Python functions with type hints and docstrings.
    - `Grader.grade` must be async and return a float in `[0.0, 1.0]`.
    - Before `osmosis train submit`, validate the project and run a local eval.

    ## Environment variables and secrets

    Training configs can inject environment variables into the rollout container via
    two optional TOML sections:

    ```toml
    [rollout.env]
    # Literal values baked into the config - visible in this file.
    # Do NOT store secrets here.
    LOG_LEVEL = "INFO"

    [rollout.secrets]
    # Maps env-var name -> workspace environment_secret *record name*.
    # The platform resolves the actual value server-side; it never appears
    # in this file or in transit.
    # Pre-register secrets at /:orgName/secrets before submitting.
    OPENAI_API_KEY = "openai-api-key"
    ```

    - Both sections are optional; omit entirely if not needed.
    - Keys must match `^[A-Z_][A-Z0-9_]*$`.
    - The same key cannot appear in both sections.
    - Reserved names (`GITHUB_CLONE_URL`, `GITHUB_TOKEN`, `ENTRYPOINT_SCRIPT`,
      `REPOSITORY_PATH`, `TRAINING_RUN_ID`, `ROLLOUT_NAME`, `ROLLOUT_PORT`) are
      forbidden in both sections.
    - Inside the container, all injected vars are available via `os.environ`.

    ## AI skills

    Detailed workflow guidance lives in the **`osmosis` agent plugin**:

    | Skill | What it does |
    | --- | --- |
    | `plan-training` | Turn a vague task into a concrete local training plan. |
    | `create-rollouts` | Create or adapt rollouts, graders, entrypoints, and baseline eval configs. |
    | `evaluate-rollouts` | Run local evals, compare baselines, and iterate with data. |
    | `debug-rollouts` | Diagnose rollout, grader, config, dataset, or preflight failures. |
    | `submit-training` | Prepare a training config and submit it safely. |

    ### Enabling the plugin

    - **Claude Code** - `.claude/settings.json` in this project registers the
      plugin automatically; on first open, Claude Code prompts to install.
    - **Cursor** - Settings -> Rules -> "Add Remote Rule" -> paste the plugin repo
      URL (skills render as Remote Rules in Cursor).
    - **Codex** - Run `codex plugin marketplace add Osmosis-AI/osmosis-plugins` once, then
      `codex plugin install osmosis`.

    The plugin repo is configured in `.claude/settings.json`. Check that file
    before modifying plugin state.

    ## CLI output

    - The commands below use the default rich output for interactive human sessions.
    - For AI agents or automation, prefer `osmosis --json ...` for structured output
      or `osmosis --plain ...` for low-noise text.

    ## Common commands

    ```bash
    osmosis project doctor
    osmosis rollout validate configs/eval/<name>.toml
    osmosis rollout validate configs/training/<run>.toml
    osmosis eval run configs/eval/<name>.toml
    osmosis train submit configs/training/<run>.toml
    osmosis train status <run-name>
    ```
    """
)

CLAUDE_MD = dedent(
    """\
    # Osmosis Project

    Before beginning any task, read `AGENTS.md` in this project.
    """
)

CONFIGS_AGENTS_MD = dedent(
    """\
    # Training & Evaluation Configs

    Configs are project-scoped and must stay in their canonical directories.

    ## Canonical paths

    - Training: `configs/training/<name>.toml`
    - Eval: `configs/eval/<name>.toml`

    Do not place these configs elsewhere. The CLI validates these locations.
    For AI agents or automation, prefer `osmosis --json ...` for structured output
    or `osmosis --plain ...` for low-noise text.

    ## Training configs (`training/*.toml`)

    Start from the default template:

    ```bash
    cp configs/training/default.toml configs/training/<run_name>.toml
    ```

    Then fill in the required fields:

    - `rollout` must match a directory under `rollouts/`
    - `entrypoint` must be a Python path relative to that rollout, usually `main.py`
    - `model_path` must be a supported base model
    - `dataset` must be a platform dataset name

    ## Eval configs (`eval/*.toml`)

    Start from the default template:

    ```bash
    cp configs/eval/default.toml configs/eval/<run_name>.toml
    ```

    Use one eval config per rollout baseline. `entrypoint` should usually be `main.py`.
    Local datasets should point at `data/*.jsonl`; use `[eval].limit` to run a
    smaller sample without creating a second dataset file.
    """
)

CLAUDE_SETTINGS_JSON = dedent(
    """\
    {
      "extraKnownMarketplaces": {
        "osmosis": {
          "source": {
            "source": "github",
            "repo": "Osmosis-AI/osmosis-plugins"
          }
        }
      },
      "enabledPlugins": {
        "osmosis@osmosis": true
      }
    }
    """
)

OFFICIAL_SCAFFOLD_FILES: tuple[OfficialScaffoldFile, ...] = (
    OfficialScaffoldFile(_path("AGENTS.md"), AGENTS_MD),
    OfficialScaffoldFile(_path("CLAUDE.md"), CLAUDE_MD),
    OfficialScaffoldFile(_path("configs/AGENTS.md"), CONFIGS_AGENTS_MD),
    OfficialScaffoldFile(_path(".claude/settings.json"), CLAUDE_SETTINGS_JSON),
)

REQUIRED_SCAFFOLD_DIRS: tuple[Path, ...] = (
    _path(".osmosis/cache"),
    _path("rollouts"),
    _path("configs/eval"),
    _path("configs/training"),
    _path("data"),
)


def recipes_by_name() -> dict[str, TemplateRecipe]:
    return {recipe.name: recipe for recipe in TEMPLATE_RECIPES}


def official_files_by_path() -> dict[str, OfficialScaffoldFile]:
    return {file.path.as_posix(): file for file in OFFICIAL_SCAFFOLD_FILES}


__all__ = [
    "OFFICIAL_SCAFFOLD_FILES",
    "REQUIRED_SCAFFOLD_DIRS",
    "ScaffoldEntry",
    "TemplateRecipe",
    "official_files_by_path",
    "recipes_by_name",
]
