"""Local rollout server example.

Usage:
    python examples/rollout/rollout_server_example.py
"""

import logging

import uvicorn
from multiply_rollout.grader import MultiplyGrader, multiply_grader_config
from multiply_rollout.workflow import MultiplyWorkflow, multiply_workflow_config

from osmosis_ai.rollout.backend.local import LocalBackend
from osmosis_ai.rollout.server import create_rollout_server

logger = logging.getLogger(__name__)


def main():
    backend = LocalBackend(
        workflow=MultiplyWorkflow,
        workflow_config=multiply_workflow_config,
        grader=MultiplyGrader,
        grader_config=multiply_grader_config,
    )
    app = create_rollout_server(backend=backend)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
