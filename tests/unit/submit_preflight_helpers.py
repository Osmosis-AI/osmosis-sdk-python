"""Shared helpers for the pinned-commit submit preflight tests.

``train submit`` and ``eval submit`` both route through
``osmosis_ai.platform.cli.shared_submit.run_cloud_submit``, so the
``check_pinned_commit`` seam and the assertions around it are identical. These
helpers let each command's test suite exercise that shared behavior without
duplicating the stubbing logic — callers only supply the command-specific
``submit`` callable and console accessor.
"""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest

import osmosis_ai.platform.cli.shared_submit as shared_submit_module
from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output import OperationResult

# Wording mirrors osmosis_ai.platform.cli.workspace_repo.check_pinned_commit so
# the assertions below stay coupled to the real user-facing messages.
REMOTE_MISSING_ERROR = (
    "Pinned commit deadbeef was not found on the connected repository (acme/rollouts)."
)
LOCAL_MISS_WARNING = "Could not find pinned commit deadbeef in the local repository."


def _stub_preflight(
    monkeypatch: pytest.MonkeyPatch,
    *,
    error: str | None,
    warnings: tuple[str, ...],
) -> None:
    monkeypatch.setattr(
        shared_submit_module,
        "check_pinned_commit",
        lambda **_kwargs: SimpleNamespace(error=error, warnings=warnings),
    )


def disable_commit_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the pinned-commit preflight a no-op (no real git/GitHub calls).

    Used by submit-test autouse fixtures so configs that set ``commit_sha`` do
    not reach out to git or GitHub during unrelated assertions.
    """
    _stub_preflight(monkeypatch, error=None, warnings=())


def assert_submit_aborts_on_invalid_commit(
    monkeypatch: pytest.MonkeyPatch,
    *,
    submit: Callable[[], Any],
) -> None:
    """A confirmed-bad pinned commit must raise before the API call.

    ``submit`` should invoke the command's submit entry point with a
    FakeClient whose submit method raises if it is ever called, so this also
    proves the preflight fails fast.
    """
    _stub_preflight(monkeypatch, error=REMOTE_MISSING_ERROR, warnings=())
    with pytest.raises(CLIError, match="was not found on the connected repository"):
        submit()


def assert_submit_surfaces_commit_warning(
    monkeypatch: pytest.MonkeyPatch,
    *,
    submit: Callable[[], OperationResult],
    console_output: Callable[[], str],
) -> None:
    """A non-blocking preflight warning must surface and still let submit run."""
    _stub_preflight(monkeypatch, error=None, warnings=(LOCAL_MISS_WARNING,))
    result = submit()
    assert isinstance(result, OperationResult)
    assert "Could not find pinned commit deadbeef" in console_output()
