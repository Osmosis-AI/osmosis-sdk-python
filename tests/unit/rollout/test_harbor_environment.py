"""Shared Harbor environment helpers: managed SkyPilot placement."""

from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import EnvironmentConfig

from osmosis_ai.rollout.backend.harbor.environment import (
    apply_managed_skypilot_placement,
)


class TestManagedSkypilotPlacement:
    """Placement comes from the run environment so rollouts name no infrastructure."""

    @staticmethod
    def _config(env_type, **kwargs):
        return EnvironmentConfig(type=env_type, kwargs=kwargs)

    def test_fills_context_name_from_environment(self, monkeypatch):
        monkeypatch.setenv("HARBOR_SKYPILOT_CONTEXT", "managed-context")
        config = self._config(EnvironmentType.SKYPILOT)

        assert (
            apply_managed_skypilot_placement(config).kwargs["context_name"]
            == "managed-context"
        )

    def test_explicit_context_name_wins(self, monkeypatch):
        monkeypatch.setenv("HARBOR_SKYPILOT_CONTEXT", "managed-context")
        config = self._config(EnvironmentType.SKYPILOT, context_name="mine")

        assert apply_managed_skypilot_placement(config).kwargs["context_name"] == "mine"

    def test_unset_environment_leaves_kwargs_untouched(self, monkeypatch):
        monkeypatch.delenv("HARBOR_SKYPILOT_CONTEXT", raising=False)
        config = self._config(EnvironmentType.SKYPILOT)

        assert apply_managed_skypilot_placement(config).kwargs == {}

    def test_ignores_non_skypilot_environments(self, monkeypatch):
        monkeypatch.setenv("HARBOR_SKYPILOT_CONTEXT", "managed-context")
        config = self._config(EnvironmentType.DAYTONA)

        assert apply_managed_skypilot_placement(config).kwargs == {}
