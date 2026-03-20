"""Harbor-based rollout server example.

The agent workflow runs on the HOST via Harbor's agent adapter.
The grader runs INSIDE the container via test.sh → grader_runner.

Requirements:
    - harbor package installed
    - Docker running

Usage:
    python examples/rollout_v2/harbor_rollout_server_example.py
"""

import logging
from pathlib import Path

import uvicorn
from harbor.orchestrators.queue import QueueOrchestrator
from multiply_rollout.grader import MultiplyGrader, multiply_grader_config
from multiply_rollout.workflow import MultiplyWorkflow, multiply_workflow_config

from osmosis_ai.rollout_v2.backend.harbor import HarborBackend
from osmosis_ai.rollout_v2.server import create_rollout_server

logger = logging.getLogger(__name__)

EXAMPLES_DIR = Path(__file__).resolve().parent
SDK_ROOT = EXAMPLES_DIR.parents[1]


def main():
    orchestrator = QueueOrchestrator(
        trial_configs=[],
        n_concurrent_trials=4,
        metrics={},
    )

    backend = HarborBackend(
        orchestrator=orchestrator,
        task_dir=EXAMPLES_DIR / "multiply_harbor_task",
        user_code_dir=EXAMPLES_DIR / "multiply_rollout",
        workflow=MultiplyWorkflow,
        workflow_config=multiply_workflow_config,
        grader=MultiplyGrader,
        grader_config=multiply_grader_config,
        _sdk_source_dir=SDK_ROOT,  # local dev only
    )

    app = create_rollout_server(backend=backend)
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
