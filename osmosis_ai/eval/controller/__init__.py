from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from osmosis_ai.eval.controller.messages import preprocess_controller_messages
    from osmosis_ai.eval.controller.state import ControllerRolloutState

__all__ = ["ControllerRolloutState", "preprocess_controller_messages"]


def __getattr__(name: str):
    if name == "ControllerRolloutState":
        from osmosis_ai.eval.controller.state import ControllerRolloutState

        return ControllerRolloutState
    if name == "preprocess_controller_messages":
        from osmosis_ai.eval.controller.messages import preprocess_controller_messages

        return preprocess_controller_messages
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
