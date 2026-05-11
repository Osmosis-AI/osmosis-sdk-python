from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from osmosis_ai.eval.controller.controller import (
        EvalController,
        EvalControllerConfig,
    )
    from osmosis_ai.eval.controller.messages import preprocess_controller_messages
    from osmosis_ai.eval.controller.server import EvalControllerServer
    from osmosis_ai.eval.controller.state import ControllerRolloutState

__all__ = [
    "ControllerRolloutState",
    "EvalController",
    "EvalControllerConfig",
    "EvalControllerServer",
    "preprocess_controller_messages",
]


def __getattr__(name: str) -> object:
    if name == "EvalController":
        from osmosis_ai.eval.controller.controller import EvalController

        return EvalController
    if name == "EvalControllerConfig":
        from osmosis_ai.eval.controller.controller import EvalControllerConfig

        return EvalControllerConfig
    if name == "ControllerRolloutState":
        from osmosis_ai.eval.controller.state import ControllerRolloutState

        return ControllerRolloutState
    if name == "EvalControllerServer":
        from osmosis_ai.eval.controller.server import EvalControllerServer

        return EvalControllerServer
    if name == "preprocess_controller_messages":
        from osmosis_ai.eval.controller.messages import preprocess_controller_messages

        return preprocess_controller_messages
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
