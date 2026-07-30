"""Tests for OsmosisClient.list_workspaces (the quickstart wizard's org picker)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from osmosis_ai.platform.api.client import OsmosisClient
from osmosis_ai.platform.api.models import WorkspaceSummary

CREDS = object()


class TestListWorkspaces:
    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_list(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {
            "workspaces": [
                {
                    "id": "org-1",
                    "name": "acme",
                    "role": "owner",
                    "has_billing_setup": True,
                    "connected_repo": {
                        "id": "repo-1",
                        "repo_full_name": "acme/acme-workspace",
                    },
                },
                {
                    "id": "org-2",
                    "name": "globex",
                    "role": "member",
                    "has_billing_setup": False,
                    "connected_repo": None,
                },
            ]
        }
        client = OsmosisClient()
        workspaces = client.list_workspaces(credentials=CREDS)

        assert workspaces == [
            WorkspaceSummary(
                id="org-1",
                name="acme",
                connected_repo_full_name="acme/acme-workspace",
            ),
            WorkspaceSummary(id="org-2", name="globex", connected_repo_full_name=None),
        ]
        args, kwargs = mock_req.call_args
        assert args[0] == "/api/cli/workspaces"
        assert "method" not in kwargs
        assert kwargs["credentials"] is CREDS
        assert kwargs["require_git_repo"] is False
        assert "git_identity" not in kwargs

    @patch("osmosis_ai.platform.api.client.platform_request")
    def test_list_empty_response(self, mock_req: MagicMock) -> None:
        mock_req.return_value = {}
        client = OsmosisClient()
        assert client.list_workspaces() == []

    def test_from_dict_tolerates_a_non_dict_connected_repo(self) -> None:
        workspace = WorkspaceSummary.from_dict(
            {"id": "org-1", "name": "acme", "connected_repo": "acme/acme-workspace"}
        )
        assert workspace.connected_repo_full_name is None
