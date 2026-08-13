"""Canonical CLI command names for registration and error-path argv fallback.

Keep this module import-light: it is loaded on every CLI error, including
unknown-command failures before handlers run. ``_register_commands`` must use
these same names so the argv fallback cannot drift from the live command tree.
"""

from __future__ import annotations

STANDALONE_QUICKSTART = "quickstart"
STANDALONE_DOCTOR = "doctor"
STANDALONE_UPGRADE = "upgrade"

GROUP_DATASET = "dataset"
GROUP_TRAIN = "train"
GROUP_MODEL = "model"
GROUP_EVAL = "eval"
GROUP_BENCHMARK = "benchmark"
GROUP_ROLLOUT = "rollout"
GROUP_TEMPLATE = "template"
GROUP_DEV = "dev"
GROUP_AUTH = "auth"
GROUP_SECRET = "secret"

STANDALONE_COMMANDS: frozenset[str] = frozenset(
    {
        STANDALONE_QUICKSTART,
        STANDALONE_DOCTOR,
        STANDALONE_UPGRADE,
    }
)

COMMAND_GROUPS: frozenset[str] = frozenset(
    {
        GROUP_DATASET,
        GROUP_TRAIN,
        GROUP_MODEL,
        GROUP_EVAL,
        GROUP_BENCHMARK,
        GROUP_ROLLOUT,
        GROUP_TEMPLATE,
        GROUP_DEV,
        GROUP_AUTH,
        GROUP_SECRET,
    }
)

# Nested groups whose leaf command is the third argv token.
THREE_TOKEN_PREFIXES: frozenset[tuple[str, str]] = frozenset(
    {
        (GROUP_BENCHMARK, "runs"),
        (GROUP_DEV, "server"),
    }
)

REMOVED_TOP_LEVEL_COMMANDS: frozenset[str] = frozenset(
    {
        "deploy",
        "deployment",
        "init",
        "link",
        "login",
        "logout",
        "undeploy",
        "unlink",
        "workspace",
        "whoami",
    }
)

REMOVED_TWO_TOKEN_COMMANDS: frozenset[tuple[str, str]] = frozenset(
    {
        (GROUP_DATASET, "delete"),
        (GROUP_MODEL, "delete"),
        (GROUP_ROLLOUT, "validate"),
        (GROUP_TRAIN, "delete"),
        (GROUP_TRAIN, "traces"),
    }
)
