"""Tests for osmosis_ai.rollout_v2.server.auth."""

from osmosis_ai.rollout_v2.server.auth import ControllerAuth


class TestControllerAuth:
    def test_repr_with_key(self):
        auth = ControllerAuth(api_key="secret")
        assert "secret" not in repr(auth)
        assert "<redacted>" in repr(auth)

    def test_repr_without_key(self):
        auth = ControllerAuth(api_key=None)
        assert "None" in repr(auth)

    def test_bearer_headers_with_key(self):
        auth = ControllerAuth(api_key="my-key")
        headers = auth.as_bearer_headers()
        assert headers == {"Authorization": "Bearer my-key"}

    def test_bearer_headers_without_key(self):
        auth = ControllerAuth(api_key=None)
        assert auth.as_bearer_headers() is None
