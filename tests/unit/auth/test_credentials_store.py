"""Tests for CredentialsStore and file-backed credential operations."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from osmosis_ai.platform.auth.credentials import (
    CredentialsStore,
    OrganizationInfo,
    UserInfo,
    WorkspaceCredentials,
    _load_store,
    delete_credentials,
    delete_workspace_credentials,
    get_active_workspace,
    get_all_workspaces,
    get_valid_credentials,
    load_credentials,
    load_workspace_credentials,
    save_credentials,
    set_active_workspace,
)


def _make_creds(
    ws_name: str = "my-workspace",
    *,
    expired: bool = False,
    token_id: str | None = None,
) -> WorkspaceCredentials:
    now = datetime.now(timezone.utc)
    delta = timedelta(hours=-1) if expired else timedelta(hours=1)
    return WorkspaceCredentials(
        access_token=f"token-{ws_name}",
        token_type="Bearer",
        expires_at=now + delta,
        user=UserInfo(id="u1", email="a@b.com", name="User"),
        organization=OrganizationInfo(id="org1", name=ws_name, role="owner"),
        created_at=now,
        token_id=token_id,
    )


# ── WorkspaceCredentials ────────────────────────────────────────────


def test_to_dict_includes_token_id_when_set() -> None:
    creds = _make_creds(token_id="tok_123")
    d = creds.to_dict()
    assert d["token_id"] == "tok_123"


def test_to_dict_excludes_token_id_when_none() -> None:
    creds = _make_creds()
    d = creds.to_dict()
    assert "token_id" not in d


def test_is_expired_true() -> None:
    creds = _make_creds(expired=True)
    assert creds.is_expired() is True


def test_is_expired_false() -> None:
    creds = _make_creds(expired=False)
    assert creds.is_expired() is False


# ── CredentialsStore ────────────────────────────────────────────────


def test_store_roundtrip() -> None:
    creds = _make_creds("ws-a")
    store = CredentialsStore(active_workspace="ws-a", workspaces={"ws-a": creds})
    d = store.to_dict()
    loaded = CredentialsStore.from_dict(d)
    assert loaded.active_workspace == "ws-a"
    assert "ws-a" in loaded.workspaces


def test_store_get_active_credentials_none() -> None:
    store = CredentialsStore(active_workspace=None, workspaces={})
    assert store.get_active_credentials() is None


def test_store_get_active_credentials_found() -> None:
    creds = _make_creds("ws-a")
    store = CredentialsStore(active_workspace="ws-a", workspaces={"ws-a": creds})
    assert store.get_active_credentials() is creds


def test_store_get_active_credentials_missing_key() -> None:
    store = CredentialsStore(active_workspace="gone", workspaces={})
    assert store.get_active_credentials() is None


# ── File operations ─────────────────────────────────────────────────


@pytest.fixture()
def _patch_creds_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """Redirect credential file operations to a temp directory."""
    config_dir = tmp_path / ".osmosis"
    config_dir.mkdir()
    creds_file = config_dir / "credentials.json"

    import osmosis_ai.platform.auth.credentials as mod

    monkeypatch.setattr(mod, "CREDENTIALS_FILE", creds_file)
    return creds_file


def test_save_and_load(
    _patch_creds_file: Path,
) -> None:
    creds = _make_creds("team-alpha")
    save_credentials(creds)

    loaded = load_credentials()
    assert loaded is not None
    assert loaded.access_token == "token-team-alpha"


def test_load_returns_none_when_no_file(
    _patch_creds_file: Path,
) -> None:
    assert load_credentials() is None


def test_load_workspace_credentials(
    _patch_creds_file: Path,
) -> None:
    save_credentials(_make_creds("ws-a"))
    save_credentials(_make_creds("ws-b"))

    assert load_workspace_credentials("ws-a") is not None
    assert load_workspace_credentials("ws-b") is not None
    assert load_workspace_credentials("ws-c") is None


def test_delete_credentials(
    _patch_creds_file: Path,
) -> None:
    save_credentials(_make_creds("ws-a"))
    assert delete_credentials() is True
    assert delete_credentials() is False
    assert load_credentials() is None


def test_delete_workspace_credentials_removes_and_switches(
    _patch_creds_file: Path,
) -> None:
    save_credentials(_make_creds("ws-a"))
    save_credentials(_make_creds("ws-b"))
    # ws-b is now active
    assert get_active_workspace() == "ws-b"

    assert delete_workspace_credentials("ws-b") is True
    # should switch to remaining workspace
    assert get_active_workspace() == "ws-a"


def test_delete_workspace_credentials_nonexistent(
    _patch_creds_file: Path,
) -> None:
    assert delete_workspace_credentials("nope") is False


def test_delete_workspace_credentials_last_workspace(
    _patch_creds_file: Path,
) -> None:
    save_credentials(_make_creds("only"))
    assert delete_workspace_credentials("only") is True
    assert get_active_workspace() is None


def test_get_valid_credentials_returns_none_expired(
    _patch_creds_file: Path,
) -> None:
    save_credentials(_make_creds("ws", expired=True))
    assert get_valid_credentials() is None


def test_get_valid_credentials_returns_creds(
    _patch_creds_file: Path,
) -> None:
    save_credentials(_make_creds("ws", expired=False))
    creds = get_valid_credentials()
    assert creds is not None


def test_get_all_workspaces(
    _patch_creds_file: Path,
) -> None:
    save_credentials(_make_creds("ws-a"))
    save_credentials(_make_creds("ws-b"))
    workspaces = get_all_workspaces()
    names = [name for name, _, _ in workspaces]
    assert "ws-a" in names
    assert "ws-b" in names
    # ws-b was saved last, so it's active
    for name, _, is_active in workspaces:
        if name == "ws-b":
            assert is_active is True


def test_get_all_workspaces_empty(
    _patch_creds_file: Path,
) -> None:
    assert get_all_workspaces() == []


def test_set_active_workspace(
    _patch_creds_file: Path,
) -> None:
    save_credentials(_make_creds("ws-a"))
    save_credentials(_make_creds("ws-b"))
    assert get_active_workspace() == "ws-b"
    assert set_active_workspace("ws-a") is True
    assert get_active_workspace() == "ws-a"


def test_set_active_workspace_nonexistent(
    _patch_creds_file: Path,
) -> None:
    assert set_active_workspace("nope") is False


def test_load_store_corrupted_file(
    _patch_creds_file: Path,
) -> None:
    _patch_creds_file.write_text("not json")
    assert _load_store() is None
