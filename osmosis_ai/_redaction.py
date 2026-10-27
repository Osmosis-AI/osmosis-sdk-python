"""Known-secret redaction before presentation or serialization."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

REDACTED = "[REDACTED]"
SECRET_FIELDS = frozenset({"value", "secret", "secret_value"})
# ponytail: fixed nesting cap, far below the recursion limit and far above any
# real diagnostic; make it a parameter if structured data ever needs more.
_MAX_DEPTH = 100
# A UTF-16 surrogate pair first, so "🔑" decodes to one character.
_JSON_ESCAPE = re.compile(
    r"\\u[dD][89abAB][0-9a-fA-F]{2}\\u[dD][c-fC-F][0-9a-fA-F]{2}"
    r'|\\u[0-9a-fA-F]{4}|\\["\\/bfnrt]'
)


def _unescape_json(text: str) -> str:
    """Decode JSON string escapes in free text, as any reader of it could."""
    return _JSON_ESCAPE.sub(lambda match: json.loads(f'"{match.group()}"'), text)


def _spellings(value: str) -> set[str]:
    """The forms a known value takes in diagnostics.

    A log reader may deliver a multiline credential one line at a time; lines
    without a letter or digit ("{", "]", blanks) are formatting, and matching
    them would erase unrelated output. Secret loaders, HTTP header parsing, and
    child processes trim surrounding whitespace. JSON encoders escape quotes,
    backslashes, control characters, and optionally non-ASCII text, so an
    embedded or undecodable JSON body carries those spellings. Python ``repr``,
    which SDK messages and container text use, escapes quotes, backslashes, and
    unprintable characters its own way and picks its quote from the whole
    string, so a value can appear with either quote escaped.
    """
    texts = {value}
    texts.update(
        line for line in value.splitlines() if any(char.isalnum() for char in line)
    )
    texts.update([text.strip() for text in texts])
    texts.discard("")
    return (
        texts
        | {
            json.dumps(text, ensure_ascii=ascii_only)[1:-1]
            for text in texts
            for ascii_only in (True, False)
        }
        # Inside single quotes, which escape "'"; double quotes need no '"'.
        | {repr(f'{text}"')[1:-2] for text in texts}
        | {repr(f"{text}'")[1:-2] for text in texts if '"' not in text}
    )


class SecretRedactor:
    """Scrub known values in every spelling listed by :func:`_spellings`."""

    def __init__(self, values: Iterable[str] = ()) -> None:
        self._sources: tuple[str, ...] = ()
        self._values: tuple[str, ...] = ()
        self.extend(values)

    @property
    def values(self) -> tuple[str, ...]:
        """Registered values as supplied, for seeding another redactor."""
        return self._sources

    def extend(self, values: Iterable[str]) -> None:
        # A whitespace-only value cannot be redacted without erasing every
        # space of the output it appears in.
        known = [value for value in values if value.strip()]
        self._sources = tuple(dict.fromkeys([*self._sources, *known]))
        sensitive = {form for value in self._sources for form in _spellings(value)}
        self._values = tuple(sorted(sensitive, key=lambda value: (-len(value), value)))

    def scrub(self, text: str) -> str:
        redacted = text
        for value in self._values:
            redacted = redacted.replace(value, REDACTED)
        # Escapes such as s can spell a value that no replacement saw,
        # whatever surrounds them; decoding shows what a reader could recover.
        decoded = _unescape_json(redacted) if "\\" in redacted else redacted
        if not any(value in redacted or value in decoded for value in self._values):
            return redacted
        # Nothing in the text is safe then. A marker that contains a supplied
        # secret, or rebuilds one with its neighbours, is not safe either.
        return "" if any(value in REDACTED for value in self._values) else REDACTED

    def scrub_lines(self, text: str) -> str:
        """Scrub line by line, so an unsafe match costs its line, not the text."""
        scrubbed = "\n".join(self.scrub(line) for line in text.split("\n"))
        # No single line holds a value that spans lines; the rejoined text can.
        if any("\n" in value and value in scrubbed for value in self._values):
            return self.scrub(scrubbed)
        return scrubbed

    def scrub_value(
        self, data: Any, *, sensitive_keys: frozenset[str] = frozenset()
    ) -> Any:
        """Copy JSON data, retaining unrelated types and sanitizing before encoding.

        A branch nested deeper than any diagnostic becomes a marker rather than
        a recursion error, which would carry the error being presented along.
        """
        if not self._values:
            return data
        return self._scrub_value(data, sensitive_keys, 0)

    def _scrub_value(
        self, data: Any, sensitive_keys: frozenset[str], depth: int
    ) -> Any:
        if depth > _MAX_DEPTH:
            return self.scrub(REDACTED)
        depth += 1
        if isinstance(data, str):
            return self.scrub(data)
        if isinstance(data, dict):
            return {
                self.scrub(key) if isinstance(key, str) else key: (
                    self.scrub(REDACTED)
                    if str(key).lower() in sensitive_keys
                    else self._scrub_value(value, sensitive_keys, depth)
                )
                for key, value in data.items()
            }
        if isinstance(data, (list, tuple)):
            return [self._scrub_value(value, sensitive_keys, depth) for value in data]
        if data is None or isinstance(data, (bool, int, float)):
            for text in (json.dumps(data), str(data)):
                if self.scrub(text) != text:
                    return self.scrub(REDACTED)
        return data


# Provider and platform keys that often live in the process environment
# without being listed in [secrets]. Skip GOOGLE_APPLICATION_CREDENTIALS:
# that value is a filesystem path, not the credential.
AMBIENT_SECRET_ENV_NAMES: frozenset[str] = frozenset(
    {
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "XAI_API_KEY",
        "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
        "OPENROUTER_API_KEY",
        "CEREBRAS_API_KEY",
        "AZURE_API_KEY",
        "TOGETHER_API_KEY",
        "FIREWORKS_API_KEY",
        "GROQ_API_KEY",
        "MISTRAL_API_KEY",
        "COHERE_API_KEY",
        "DEEPSEEK_API_KEY",
        "HUGGINGFACE_API_KEY",
        "HF_TOKEN",
        "WANDB_API_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "OSMOSIS_TOKEN",
        "OSMOSIS_API_KEY",
        "DAYTONA_API_KEY",
        "CURSOR_API_KEY",
        "MSWEA_API_KEY",
    }
)
MIN_AMBIENT_SECRET_LEN = 8


def ambient_secret_values(env: Mapping[str, str]) -> list[str]:
    """Non-empty ambient env values long enough to redact without collateral."""
    values: list[str] = []
    for name in AMBIENT_SECRET_ENV_NAMES:
        value = env.get(name)
        if value and len(value) >= MIN_AMBIENT_SECRET_LEN:
            values.append(value)
    return values


_invocation_redactor: ContextVar[SecretRedactor | None] = ContextVar(
    "invocation_redactor", default=None
)


@contextmanager
def invocation_redaction() -> Iterator[SecretRedactor]:
    """Collect the secrets one CLI invocation handles, for its error output."""
    token = _invocation_redactor.set(SecretRedactor())
    try:
        yield invocation_redactor()
    finally:
        _invocation_redactor.reset(token)


def invocation_redactor() -> SecretRedactor:
    """The current invocation's redactor; an empty one outside an invocation."""
    return _invocation_redactor.get() or SecretRedactor()


def register_invocation_secrets(values: Iterable[str]) -> None:
    """Make *values* known to the current CLI invocation's error boundary."""
    redactor = _invocation_redactor.get()
    if redactor is not None:
        redactor.extend(values)
