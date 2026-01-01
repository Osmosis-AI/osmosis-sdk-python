from __future__ import annotations

from datetime import datetime, timedelta, timezone

from osmosis_ai.auth.credentials import WorkspaceCredentials, OrganizationInfo, UserInfo


def _make_credentials(
    *,
    expires_at: datetime,
    created_at: datetime,
) -> WorkspaceCredentials:
    return WorkspaceCredentials(
        access_token="test-token",
        token_type="Bearer",
        expires_at=expires_at,
        user=UserInfo(id="user_1", email="user@example.com", name="User"),
        organization=OrganizationInfo(id="org_1", name="Org", role="member"),
        created_at=created_at,
    )


def test_credentials_roundtrip_preserves_tz_aware_expires_at() -> None:
    now_utc = datetime.now(timezone.utc)
    creds = _make_credentials(
        expires_at=now_utc + timedelta(minutes=5),
        created_at=now_utc,
    )

    data = creds.to_dict()
    loaded = WorkspaceCredentials.from_dict(data)

    assert loaded.expires_at.tzinfo is not None
    assert loaded.is_expired() is False


def test_from_dict_rejects_naive_expires_at() -> None:
    now = datetime.now(timezone.utc)
    creds = _make_credentials(
        expires_at=now + timedelta(minutes=5),
        created_at=now,
    )

    data = creds.to_dict()
    data["expires_at"] = datetime.now().isoformat()  # naive, no tz offset

    try:
        WorkspaceCredentials.from_dict(data)
    except ValueError as exc:
        assert "expires_at must be timezone-aware" in str(exc)
    else:
        raise AssertionError("Expected ValueError for naive expires_at")


