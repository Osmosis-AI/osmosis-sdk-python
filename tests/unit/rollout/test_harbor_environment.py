"""Shared Harbor environment helpers: loopback routing."""


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
            "http://[::ffff:127.0.0.1]:8080/v1",
        ):
            assert (
                env_module.rewrite_url_for_docker(url)
                == "http://host.docker.internal:8080/v1"
            ), url
        assert (
            env_module.rewrite_url_for_docker("http://user:pass@localhost:8080/v1")
            == "http://user:pass@host.docker.internal:8080/v1"
        )

    def test_macos_rewrite_leaves_public_hosts_alone(self, monkeypatch):
        from osmosis_ai.rollout.backend.harbor import environment as env_module

        monkeypatch.setattr(env_module.platform, "system", lambda: "Darwin")
        url = "https://name.trycloudflare.com/v1/rollouts/r1"
        assert env_module.rewrite_url_for_docker(url) == url
