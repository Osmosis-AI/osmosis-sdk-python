"""Regression tests for workspace-specific 401 cleanup."""

from __future__ import annotations

import http.client
from datetime import datetime, timedelta, timezone
from io import BytesIO
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError

import pytest

from osmosis_ai.platform.auth.credentials import (
    OrganizationInfo,
    UserInfo,
    WorkspaceCredentials,
)
from osmosis_ai.platform.auth.platform_client import (
    AuthenticationExpiredError,
    platform_request,
)


def _make_credentials(org_name: str) -> WorkspaceCredentials:
    now = datetime.now(timezone.utc)
    return WorkspaceCredentials(
        access_token="test-token-abc123",
        token_type="Bearer",
        expires_at=now + timedelta(days=30),
        user=UserInfo(id="user_1", email="user@example.com", name="Test User"),
        organization=OrganizationInfo(id="org_1", name=org_name, role="member"),
        created_at=now,
    )


def _make_http_error(
    code: int, body: str = "", url: str = "https://platform.osmosis.ai/api/test"
) -> HTTPError:
    fp = BytesIO(body.encode("utf-8")) if body else BytesIO(b"")
    return HTTPError(
        url=url,
        code=code,
        msg=http.client.responses.get(code, "Error"),
        hdrs={},
        fp=fp,
    )


@patch("osmosis_ai.platform.auth.platform_client.delete_workspace_credentials")
@patch("osmosis_ai.platform.auth.platform_client.urlopen")
def test_platform_request_cleans_up_explicit_workspace_on_401(
    mock_urlopen: MagicMock,
    mock_delete_workspace_credentials: MagicMock,
) -> None:
    mock_urlopen.side_effect = _make_http_error(401, "Unauthorized")

    with pytest.raises(AuthenticationExpiredError):
        platform_request("/api/test", credentials=_make_credentials("ws-b"))

    mock_delete_workspace_credentials.assert_called_once_with("ws-b")
