"""Local OpenAI Agents multiply rollout server."""

from __future__ import annotations

import os

import uvicorn

from multiply_rollout.grader import MultiplyGrader, multiply_grader_config
from multiply_rollout.workflow import MultiplyWorkflow, multiply_workflow_config
from osmosis_ai.rollout.backend.local import LocalBackend
from osmosis_ai.rollout.server import create_rollout_server


def main() -> None:
    backend = LocalBackend(
        workflow=MultiplyWorkflow,
        workflow_config=multiply_workflow_config,
        grader=MultiplyGrader,
        grader_config=multiply_grader_config,
    )
    app = create_rollout_server(backend=backend)
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("_OSMOSIS_ROLLOUT_PORT", "8000")))


if __name__ == "__main__":
    main()
