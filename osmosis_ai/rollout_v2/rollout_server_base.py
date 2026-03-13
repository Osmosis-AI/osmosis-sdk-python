import logging
import traceback
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException

from osmosis_ai.rollout_v2.execution_backend import ExecutionBackend
from osmosis_ai.rollout_v2.types import (
    RolloutInitRequest,
    RolloutInitResponse,
)

logger = logging.getLogger(__name__)


def create_app(*, backend: ExecutionBackend) -> FastAPI:
    app = FastAPI()

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return backend.health()

    @app.post("/rollout")
    async def rollout(
        request: RolloutInitRequest, background_tasks: BackgroundTasks
    ) -> RolloutInitResponse:
        try:
            background_tasks.add_task(backend.execute_rollout, request)
            return RolloutInitResponse()
        except Exception as e:
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e)) from e

    return app
