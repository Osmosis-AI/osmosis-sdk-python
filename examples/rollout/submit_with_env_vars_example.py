"""Programmatic run-launch example: submitting with rollout_env and rollout_secret_refs.

This script demonstrates how to submit a training run with literal environment
variables and workspace secret references directly through the OsmosisClient
Python API, bypassing the TOML-based CLI flow.  Useful for CI pipelines,
programmatic tooling, or scripts that build training configs at runtime.

Key concepts
------------
rollout_env
    A dict of env-var-name → literal string value.  These are injected verbatim
    into the rollout container.  Because the values appear in this file and in
    the API payload, do NOT store secrets here.

rollout_secret_refs
    A dict of env-var-name → workspace ``environment_secret`` *record name*.
    Values are resolved server-side from the workspace's encrypted secret store
    and injected into the rollout container at launch.  Secret values never
    appear in this file, in transit, or in CLI output.

Reserved env-var names (cannot be used in either dict):
    GITHUB_CLONE_URL, GITHUB_TOKEN, ENTRYPOINT_SCRIPT, REPOSITORY_PATH,
    TRAINING_RUN_ID, ROLLOUT_NAME, ROLLOUT_PORT

Prerequisites
-------------
1. ``osmosis auth login``  (or export OSMOSIS_TOKEN=<your-token>)
2. The workspace environment_secret named by ``--secret-name`` must be
   pre-registered at /:orgName/secrets in the Osmosis platform UI.
3. The "multiply" rollout must exist in the workspace's connected repository.
   Run from the osmosis-workspace-internal project root so the CLI can resolve
   the workspace, or pass an explicit project root if running from elsewhere.

Usage
-----
    # From workspace root (osmosis-workspace-internal):
    cd osmosis-workspace-internal
    python ../osmosis-sdk-python/examples/rollout/submit_with_env_vars_example.py

    # Dry-run: print the payload without submitting:
    python submit_with_env_vars_example.py --dry-run

    # Override the secret record name:
    python submit_with_env_vars_example.py --secret-name my-openai-key
"""

from __future__ import annotations

import argparse
import json
import sys

from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.auth.credentials import load_credentials


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the payload that would be submitted without actually submitting.",
    )
    parser.add_argument(
        "--secret-name",
        default="openai-api-key",
        metavar="NAME",
        help=(
            "Name of the workspace environment_secret to inject as OPENAI_API_KEY.  "
            "Must be pre-registered at /:orgName/secrets.  (default: openai-api-key)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Literal env vars — injected verbatim; values are visible in this file.
    rollout_env: dict[str, str] = {
        "LOG_LEVEL": "INFO",
        "DEFAULT_REGION": "us-west-2",
    }

    # Secret references — env-var name → workspace environment_secret record name.
    # The platform resolves the actual secret value server-side; it never appears
    # in this file, in the API request body, or in any CLI output.
    rollout_secret_refs: dict[str, str] = {
        "OPENAI_API_KEY": args.secret_name,
    }

    print("Training run parameters:")
    print("  model_path:          Qwen/Qwen3.6-35B-A3B")
    print("  dataset:             multiply_1k-2026-04-13-66b645")
    print("  rollout:             multiply")
    print("  entrypoint:          local_rollout_server_with_env_example.py")
    print(f"  rollout_env:         {rollout_env}")
    print(f"  rollout_secret_refs: {rollout_secret_refs}")
    print()

    if args.dry_run:
        payload = {
            "model_path": "Qwen/Qwen3.6-35B-A3B",
            "dataset": "multiply_1k-2026-04-13-66b645",
            "rollout_name": "multiply",
            "entrypoint": "local_rollout_server_with_env_example.py",
            "rollout_env": rollout_env,
            "rollout_secret_refs": rollout_secret_refs,
        }
        print("Dry-run — payload that would be sent to /api/cli/training-runs:")
        print(json.dumps(payload, indent=2))
        return

    credentials = load_credentials()
    if credentials is None:
        print(
            "Not authenticated. Run: osmosis auth login",
            file=sys.stderr,
        )
        sys.exit(1)

    client = OsmosisClient()
    result = client.submit_training_run(
        model_path="Qwen/Qwen3.6-35B-A3B",
        dataset="multiply_1k-2026-04-13-66b645",
        rollout_name="multiply",
        entrypoint="local_rollout_server_with_env_example.py",
        rollout_env=rollout_env,
        rollout_secret_refs=rollout_secret_refs,
        credentials=credentials,
    )

    print(f"Submitted: {result.name}")
    print(f"Status:    {result.status}")
    print(f"ID:        {result.id}")
    print()
    print(f"Check status:  osmosis train status {result.name}")
    print(f"Poll until done:  osmosis --json train status {result.name}")


if __name__ == "__main__":
    main()
