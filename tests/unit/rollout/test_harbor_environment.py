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


class TestLoopbackUrlClassification:
    """The guard's predicate and the Docker rewrite must classify alike."""

    def test_every_loopback_form_is_loopback(self):
        from osmosis_ai.rollout.backend.harbor.environment import is_loopback_url

        for url in (
            "http://127.0.0.1:8080/v1",
            "http://localhost:8080/v1",
            "http://[::1]:8080/v1",
            # IPv4-mapped IPv6 loopback; is_loopback alone misses it.
            "http://[::ffff:127.0.0.1]:8080/v1",
        ):
            assert is_loopback_url(url), url

    def test_public_and_empty_hosts_are_not_loopback(self):
        from osmosis_ai.rollout.backend.harbor.environment import is_loopback_url

        assert not is_loopback_url("https://eval.example.com/v1")
        assert not is_loopback_url("")

    def test_macos_rewrite_covers_every_loopback_form(self, monkeypatch):
        from osmosis_ai.rollout.backend.harbor import environment as env_module

        monkeypatch.setattr(env_module.platform, "system", lambda: "Darwin")
        for url in (
            "http://127.0.0.1:8080/v1",
            "http://localhost:8080/v1",
            "http://[::1]:8080/v1",
        ):
            assert (
                env_module.rewrite_url_for_docker(url)
                == "http://host.docker.internal:8080/v1"
            ), url

    def test_macos_rewrite_leaves_public_hosts_alone(self, monkeypatch):
        from osmosis_ai.rollout.backend.harbor import environment as env_module

        monkeypatch.setattr(env_module.platform, "system", lambda: "Darwin")
        url = "https://name.trycloudflare.com/v1/rollouts/r1"
        assert env_module.rewrite_url_for_docker(url) == url
