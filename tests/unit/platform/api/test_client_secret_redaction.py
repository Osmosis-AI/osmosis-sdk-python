"""Each secret-bearing endpoint supplies redaction context to the real transport."""

from __future__ import annotations

import json
from io import BytesIO
from types import SimpleNamespace
from urllib.error import HTTPError

import pytest

from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.auth.platform_client import PlatformAPIError

SECRET = 'synth-key-A"B\\C-123456789'


@pytest.fixture(
    params=[
        lambda client, kwargs, secret: client.set_environment_secret(
            "TOKEN", secret, scope="user", **kwargs
        ),
        lambda client, kwargs, secret: client.submit_training_run(
            experiment_config={}, provided_secrets={"TOKEN": secret}, **kwargs
        ),
        lambda client, kwargs, secret: client.submit_evaluation_run(
            experiment_config={}, provided_secrets={"TOKEN": secret}, **kwargs
        ),
        lambda client, kwargs, secret: client.submit_benchmark_run(
            experiment_config={},
            agents=[],
            secrets={"provided": {"TOKEN": secret}},
            **kwargs,
        ),
        lambda client, kwargs, secret: client.retry_eval_run(
            "run-1", secrets={"TOKEN": secret}, **kwargs
        ),
    ],
    ids=["secret-set", "train", "eval", "benchmark", "eval-retry"],
)
def submit(request):
    return request.param


def test_secret_endpoints_sanitize_decoded_http_errors(monkeypatch, submit) -> None:
    import osmosis_ai.platform.auth.platform_client as transport

    monkeypatch.setattr(transport, "get_platform_url", lambda: "https://example.test")

    def reject(request, **kwargs):
        assert json.dumps(SECRET)[1:-1] in request.data.decode()
        body = json.dumps({"received": [SECRET], "safe": 42}).encode()
        raise HTTPError(request.full_url, 400, "Bad Request", {}, BytesIO(body))

    monkeypatch.setattr(transport, "urlopen", reject)
    with pytest.raises(PlatformAPIError) as caught:
        submit(
            OsmosisClient(),
            {
                "credentials": SimpleNamespace(access_token="synthetic-auth"),
                "git_identity": "acme/rollout",
            },
            SECRET,
        )

    assert caught.value.status_code == 400
    assert caught.value.details == {"received": ["[REDACTED]"], "safe": 42}
    assert SECRET not in str(caught.value)
    assert json.dumps(SECRET)[1:-1] not in str(caught.value)


@pytest.mark.parametrize("secret", ["id", "1", SECRET])
def test_secret_endpoints_preserve_success_protocol_before_presentation(
    monkeypatch, submit, secret
):
    from unittest.mock import MagicMock

    import osmosis_ai.platform.auth.platform_client as transport

    body = {
        "id": "run-id",
        "name": secret,
        "status": "pending",
        "created_at": "now",
        "updated_at": "now",
        "workflow_id": "workflow-id",
        "retryable_samples": 1,
        "task_count": 1,
        "platform_url": "https://example.test/run",
    }
    response = MagicMock()
    response.__enter__.return_value = response
    response.status = 200
    response.headers = {}
    response.read.return_value = json.dumps(body).encode()
    monkeypatch.setattr(transport, "urlopen", lambda *args, **kwargs: response)
    monkeypatch.setattr(transport, "get_platform_url", lambda: "https://example.test")
    result = submit(
        OsmosisClient(),
        {
            "credentials": SimpleNamespace(access_token="synthetic-auth"),
            "git_identity": "acme/rollout",
        },
        secret,
    )
    assert result.id == "run-id"
    assert result.name == secret
    if hasattr(result, "retryable_samples"):
        assert result.retryable_samples == 1


@pytest.mark.parametrize("malformed", ["missing-or-count", "name", "platform_url"])
def test_malformed_success_never_exposes_parser_exception_values(
    monkeypatch, submit, malformed
):
    import traceback
    from unittest.mock import MagicMock

    import osmosis_ai.platform.auth.platform_client as transport

    def respond(request, **kwargs):
        # Missing required keys, or an invalid numeric leaf in the retry schema.
        body = (
            {
                "id": "run-id",
                "name": "run",
                "status": "pending",
                "workflow_id": "workflow",
                "retryable_samples": SECRET,
            }
            if request.full_url.endswith("/retry")
            else {}
        )
        if malformed != "missing-or-count":
            body = {
                "id": "run-id",
                "name": "run",
                "status": "pending",
                "created_at": "now",
                "workflow_id": "workflow",
                "task_count": 1,
                "retryable_samples": 1,
            }
            body[malformed] = [SECRET] if malformed == "name" else {"echo": SECRET}
        response = MagicMock()
        response.__enter__.return_value = response
        response.status = 200
        response.headers = {}
        response.read.return_value = json.dumps(body).encode()
        return response

    monkeypatch.setattr(transport, "urlopen", respond)
    monkeypatch.setattr(transport, "get_platform_url", lambda: "https://example.test")
    with pytest.raises(
        PlatformAPIError, match="Invalid response from platform"
    ) as caught:
        submit(
            OsmosisClient(),
            {
                "credentials": SimpleNamespace(access_token="synthetic-auth"),
                "git_identity": "acme/rollout",
            },
            SECRET,
        )
    assert SECRET not in "".join(traceback.format_exception(caught.value))
    assert caught.value.__suppress_context__
