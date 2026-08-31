from __future__ import annotations

from pathlib import Path

import pytest

from osmosis_ai.cli.errors import CLIError
from osmosis_ai.cli.output.context import OutputFormat, override_output_context
from osmosis_ai.platform.cli.secret_resolution import resolve_run_secrets


def test_secrets_file_wins_over_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("A_KEY", "from-env")
    path = tmp_path / ".env"
    path.write_text("A_KEY=from-file\n", encoding="utf-8")

    assert resolve_run_secrets(
        names=["A_KEY"], secrets_file=str(path), stored_names=set()
    ) == {"A_KEY": "from-file"}


def test_environment_used_when_no_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("A_KEY", "from-env")

    assert resolve_run_secrets(
        names=["A_KEY"], secrets_file=None, stored_names=set()
    ) == {"A_KEY": "from-env"}


def test_stored_names_are_left_to_the_server(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("A_KEY", raising=False)

    assert (
        resolve_run_secrets(names=["A_KEY"], secrets_file=None, stored_names={"A_KEY"})
        == {}
    )


def test_non_tty_lists_every_missing_name_at_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("A_KEY", raising=False)
    monkeypatch.delenv("B_KEY", raising=False)

    with override_output_context(format=OutputFormat.rich, interactive=False):
        with pytest.raises(CLIError, match=r"A_KEY.*B_KEY") as exc:
            resolve_run_secrets(
                names=["A_KEY", "B_KEY"],
                secrets_file=None,
                stored_names=set(),
            )

    assert exc.value.code == "INTERACTIVE_REQUIRED"
    assert exc.value.details == {
        "missing": ["A_KEY", "B_KEY"],
        "flags": ["--secrets-file"],
    }
    assert "--secrets-file" in exc.value.message
    assert "osmosis secret set" in exc.value.message


def test_comments_and_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("# a comment\n\nA_KEY=value\n", encoding="utf-8")

    assert resolve_run_secrets(
        names=["A_KEY"], secrets_file=str(path), stored_names=set()
    ) == {"A_KEY": "value"}


def test_a_malformed_secrets_file_line_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text("NOT_AN_ASSIGNMENT\n", encoding="utf-8")

    with pytest.raises(CLIError, match="Invalid line"):
        resolve_run_secrets(
            names=["A_KEY"],
            secrets_file=str(path),
            stored_names=set(),
        )


SENTINEL = "sk-live-do-not-print-me"


@pytest.mark.parametrize(
    "content",
    [
        # A value pasted without its NAME= prefix.
        f"{SENTINEL}\n",
        # The continuation of a multi-line value — a pasted PEM key being the
        # realistic case. The opening line parses; the body that follows has
        # no '=' and is what used to land in the error. (Spelled without a
        # real PEM header so the pre-commit key scanner stays useful here.)
        f'SIGNING_KEY="--BEGIN TEST KEY--\n{SENTINEL}\n',
        # A name that is not a plain identifier: rejected, still not echoed.
        f"not a name={SENTINEL}\n",
    ],
)
def test_errors_never_quote_the_offending_line(tmp_path: Path, content: str) -> None:
    path = tmp_path / ".env"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(CLIError) as exc:
        resolve_run_secrets(names=["A_KEY"], secrets_file=str(path), stored_names=set())

    rendered = f"{exc.value} {exc.value.message} {exc.value.details}"
    assert SENTINEL not in rendered
    assert str(path) in exc.value.message


def test_stdin_is_named_rather_than_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO(f"{SENTINEL}\n"))

    with pytest.raises(CLIError) as exc:
        resolve_run_secrets(names=["A_KEY"], secrets_file="-", stored_names=set())

    assert SENTINEL not in exc.value.message
    assert "stdin" in exc.value.message


def test_the_reported_line_number_locates_the_bad_line(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(f"# note\n\nA_KEY=ok\n{SENTINEL}\n", encoding="utf-8")

    with pytest.raises(CLIError, match="Invalid line 4"):
        resolve_run_secrets(names=["A_KEY"], secrets_file=str(path), stored_names=set())


@pytest.mark.parametrize(
    "line,expected",
    [
        ('A_KEY="quoted"', "quoted"),
        ("A_KEY='quoted'", "quoted"),
        ("export A_KEY=exported", "exported"),
        ('export A_KEY="both"', "both"),
        ("A_KEY=", ""),
        ('A_KEY=un"matched', 'un"matched'),
        ("A_KEY=has=equals", "has=equals"),
    ],
)
def test_common_dotenv_forms_resolve_to_the_intended_value(
    tmp_path: Path, line: str, expected: str
) -> None:
    # These used to be accepted silently and submit the wrong secret: quotes
    # travelled with the value, and `export A_KEY` became the name.
    path = tmp_path / ".env"
    path.write_text(line + "\n", encoding="utf-8")

    assert resolve_run_secrets(
        names=["A_KEY"], secrets_file=str(path), stored_names=set()
    ) == {"A_KEY": expected}


def _fail_if_getpass(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    def _boom(prompt: str = "") -> str:
        calls.append(prompt)
        raise AssertionError(f"getpass must not be called: {prompt}")

    monkeypatch.setattr(
        "osmosis_ai.platform.cli.secret_resolution.getpass",
        _boom,
    )
    return calls


def test_json_mode_does_not_getpass_even_when_tty_is_claimed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`--json` on a developer TTY must not dead-end on a password prompt."""
    monkeypatch.delenv("A_KEY", raising=False)
    monkeypatch.delenv("B_KEY", raising=False)
    calls = _fail_if_getpass(monkeypatch)

    with override_output_context(format=OutputFormat.json, interactive=False):
        with pytest.raises(CLIError) as exc:
            resolve_run_secrets(
                names=["A_KEY", "B_KEY"],
                secrets_file=None,
                stored_names=set(),
            )

    assert calls == []
    err = exc.value
    assert err.code == "INTERACTIVE_REQUIRED"
    assert "A_KEY" in err.message
    assert "B_KEY" in err.message
    assert "--secrets-file" in err.message
    assert err.details == {
        "missing": ["A_KEY", "B_KEY"],
        "flags": ["--secrets-file"],
    }


def test_plain_mode_does_not_getpass_even_when_tty_is_claimed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("A_KEY", raising=False)
    calls = _fail_if_getpass(monkeypatch)

    with override_output_context(format=OutputFormat.plain, interactive=False):
        with pytest.raises(CLIError) as exc:
            resolve_run_secrets(
                names=["A_KEY"],
                secrets_file=None,
                stored_names=set(),
            )

    assert calls == []
    assert exc.value.code == "INTERACTIVE_REQUIRED"


def test_empty_prompted_value_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("A_KEY", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.secret_resolution.getpass",
        lambda prompt="": "",
    )

    with override_output_context(format=OutputFormat.rich, interactive=True):
        with pytest.raises(CLIError, match="empty") as exc:
            resolve_run_secrets(
                names=["A_KEY"],
                secrets_file=None,
                stored_names=set(),
            )

    assert exc.value.code == "VALIDATION"


def test_interactive_rich_prompts_for_missing_secret(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("A_KEY", raising=False)
    monkeypatch.setattr(
        "osmosis_ai.platform.cli.secret_resolution.getpass",
        lambda prompt="": "typed-secret",
    )

    with override_output_context(format=OutputFormat.rich, interactive=True):
        assert resolve_run_secrets(
            names=["A_KEY"],
            secrets_file=None,
            stored_names=set(),
        ) == {"A_KEY": "typed-secret"}
