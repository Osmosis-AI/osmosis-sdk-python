"""Save finished rollouts as ATIF trajectory documents (backend-agnostic)."""

from typing import TYPE_CHECKING

from osmosis_ai._imports import public_module_dir, resolve_lazy_export

if TYPE_CHECKING:
    from osmosis_ai.rollout.trajectory.converter import convert_sample_to_trajectory
    from osmosis_ai.rollout.trajectory.report import (
        TrajectoryReport,
        report_from_response,
    )
    from osmosis_ai.rollout.trajectory.save import save_trajectories

_EXPORTS: dict[str, tuple[str, str]] = {
    "TrajectoryReport": (
        "osmosis_ai.rollout.trajectory.report",
        "TrajectoryReport",
    ),
    "convert_sample_to_trajectory": (
        "osmosis_ai.rollout.trajectory.converter",
        "convert_sample_to_trajectory",
    ),
    "report_from_response": (
        "osmosis_ai.rollout.trajectory.report",
        "report_from_response",
    ),
    "save_trajectories": (
        "osmosis_ai.rollout.trajectory.save",
        "save_trajectories",
    ),
}


def __getattr__(name: str) -> object:
    return resolve_lazy_export(
        name,
        module_name=__name__,
        namespace=globals(),
        exports=_EXPORTS,
    )


def __dir__() -> list[str]:
    return public_module_dir(globals(), _EXPORTS)


__all__ = [
    "TrajectoryReport",
    "convert_sample_to_trajectory",
    "report_from_response",
    "save_trajectories",
]
