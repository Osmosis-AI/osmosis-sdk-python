"""Tests for OsmosisClient quickstart methods.

Covers the wizard-facing API:
    get_quickstart_status / complete_quickstart
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.api.models import QuickstartStatus

ORG_ID = "3f1c9a52-8f2b-4d6e-9c1a-2b7d5e0f4a11"
CREDS = object()


class TestGetQuickstartStatus:
    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_get(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "repo": {"connected": True, "full_name": "acme/acme-workspace"},
            "billing_ready": True,
            "completed": False,
        }
        client = OsmosisClient()
        result = client.get_quickstart_status(ORG_ID, credentials=CREDS)
        assert isinstance(result, QuickstartStatus)
        assert result.repo_connected is True
        assert result.repo_full_name == "acme/acme-workspace"
        assert result.billing_ready is True
        assert result.completed is False
        args, kwargs = mock_req.call_args
        assert args[0] == f"/api/cli/quickstart?organizationId={ORG_ID}"
        assert "method" not in kwargs
        assert kwargs["credentials"] is CREDS
        assert kwargs["require_git_repo"] is False
        assert "git_identity" not in kwargs

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_get_unconnected_repo(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "repo": {"connected": False, "full_name": None},
            "billing_ready": False,
            "completed": False,
        }
        client = OsmosisClient()
        result = client.get_quickstart_status(ORG_ID)
        assert result.repo_connected is False
        assert result.repo_full_name is None
        assert result.billing_ready is False
        assert result.completed is False

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_get_urlencodes_organization_id(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "repo": {"connected": False, "full_name": None},
            "billing_ready": False,
            "completed": False,
        }
        client = OsmosisClient()
        client.get_quickstart_status("../bad org")
        args, _kwargs = mock_req.call_args
        assert "../" not in args[0]
        assert " " not in args[0]


class TestQuickstartStatusFromDict:
    def test_missing_repo_object(self) -> None:
        status = QuickstartStatus.from_dict({"billing_ready": True})
        assert status.repo_connected is False
        assert status.repo_full_name is None
        assert status.billing_ready is True
        assert status.completed is False

    def test_full_name_ignored_when_not_a_string(self) -> None:
        status = QuickstartStatus.from_dict(
            {"repo": {"connected": True, "full_name": 42}}
        )
        assert status.repo_connected is True
        assert status.repo_full_name is None


class TestCompleteQuickstart:
    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_post(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {"completed": True}
        client = OsmosisClient()
        result = client.complete_quickstart(ORG_ID, "train", credentials=CREDS)
        assert result is None
        args, kwargs = mock_req.call_args
        assert args[0] == "/api/cli/quickstart"
        assert kwargs["method"] == "POST"
        assert kwargs["data"] == {"organizationId": ORG_ID, "intent": "train"}
        assert kwargs["credentials"] is CREDS
        assert kwargs["require_git_repo"] is False
        assert "git_identity" not in kwargs

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_post_passes_intent_through(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {"completed": True}
        client = OsmosisClient()
        client.complete_quickstart(ORG_ID, "explore")
        _args, kwargs = mock_req.call_args
        assert kwargs["data"]["intent"] == "explore"
