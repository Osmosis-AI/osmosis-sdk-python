"""Representation-independent contracts for known-secret redaction."""

from __future__ import annotations

import ast
import contextlib
import json
import re
from itertools import combinations, product

from osmosis_ai._redaction import SecretRedactor
from tests.unit.secret_spelling_helpers import every_unit_escaped, recoverable


def test_all_secret_orders_and_overlaps_leave_clean_idempotent_strings() -> None:
    words = ["".join(chars) for size in (1, 2) for chars in product("]xy", repeat=size)]
    texts = [
        "".join(chars) for size in range(5) for chars in product("]xy", repeat=size)
    ]
    pairs = [*combinations(words, 2), ("RE", "[REDACTED]"), ("x\ny", "]x")]
    for values in pairs:
        redactor = SecretRedactor(["", *values, *values])
        reverse = SecretRedactor(reversed(values))
        for text in [*texts, *values, "[REDACTED]", "before x\ny after"]:
            output = redactor.scrub(text)
            for value in values:
                assert value not in output
                assert all(line not in output for line in value.splitlines() if line)
            assert redactor.scrub(output) == output
            assert reverse.scrub(text) == output


def test_multiline_formatting_lines_do_not_erase_unrelated_output() -> None:
    array = '[\n  "synth-key-one",\n  "synth-key-two"\n]'
    obj = '{\n  "type": "synth-account",\n \t\n}'
    redactor = SecretRedactor([array, obj])
    unrelated = 'INFO [result] done {"status": "ok"} [x]'
    assert redactor.scrub(unrelated) == unrelated
    assert redactor.scrub(f"echo {array}") == "echo [REDACTED]"
    assert redactor.scrub('line:  "synth-key-two"') == "line:[REDACTED]"
    assert redactor.scrub('  "type": "synth-account",') == "[REDACTED]"


def test_trimmed_and_json_escaped_spellings_are_redacted() -> None:
    padded = ' synth-key-A"B\\C-é-123 '
    multiline = 'first-line-é"x\n  second-line\\y  '
    redactor = SecretRedactor([padded, multiline])
    spellings = [
        padded,
        padded.strip(),
        json.dumps(padded)[1:-1],
        json.dumps(padded.strip(), ensure_ascii=False)[1:-1],
        multiline,
        json.dumps(multiline)[1:-1],
        "second-line\\y",
        json.dumps('first-line-é"x')[1:-1],
    ]
    for spelling in spellings:
        assert redactor.scrub(f"<{spelling}>") == "<[REDACTED]>"
    assert redactor.values == (padded, multiline)


def test_escapes_that_decode_to_a_secret_make_the_text_unsafe() -> None:
    secret = "synth-key-123456"
    redactor = SecretRedactor([secret, "a/b-secret-99"])
    every_char = "".join(f"\\u{ord(char):04X}" for char in secret)
    for text in (
        f'[0, "{every_char}", [[[',
        "echo synth\\u002Dkey-123456",
        "path a\\/b-secret-99",
    ):
        assert redactor.scrub(text) == "[REDACTED]"
    # An astral character escapes as a surrogate pair, which decodes as one.
    keyed = "synth-\U0001f511-key-123456"
    pair_escaped = every_unit_escaped(keyed)
    assert "\\uD83D\\uDD11" in pair_escaped
    assert recoverable(pair_escaped) == keyed
    assert SecretRedactor([keyed]).scrub(f"echo {pair_escaped}") == "[REDACTED]"
    # Backslashes that decode to nothing known keep the text readable.
    plain = "[Errno 61] C:\\new\\table"
    assert redactor.scrub(plain) == plain
    assert redactor.scrub_lines(f"kept\nuses {secret}\nkept") == (
        "kept\nuses [REDACTED]\nkept"
    )


def test_json_redaction_preserves_safe_types_and_does_not_mutate_input() -> None:
    value = {"text": "context", "number": 42, "float": 1.25, "bool": True, "null": None}
    data = {"safe": value, "nested": [{"synth-secret-key": "synth-secret"}]}
    redactor = SecretRedactor()
    assert redactor.scrub_value(data) == data
    redactor.extend(["synth-secret"])
    result = redactor.scrub_value(data)
    assert result == {"safe": value, "nested": [{"[REDACTED]-key": "[REDACTED]"}]}
    assert data["nested"] == [{"synth-secret-key": "synth-secret"}]
    assert result["safe"] is not value


def _literals(text: str) -> list[object]:
    """Every Python string literal in *text*, parsed by the standard library."""
    found = []
    for match in re.finditer(r"'(?:[^'\\]|\\.)*'|\"(?:[^\"\\]|\\.)*\"", text):
        with contextlib.suppress(ValueError, SyntaxError):
            found.append(ast.literal_eval(match.group()))
    return found


def test_python_repr_cannot_respell_a_known_value() -> None:
    # Both quotes, a backslash, and a control character: repr escapes each of
    # them differently from JSON, and picks its quote from the whole string.
    secret = "synth'\"\\\x07-key-123456"
    apostrophe = "synth-o'key-123456"
    redactor = SecretRedactor([secret, apostrophe])
    for value in (secret, apostrophe):
        for text in (
            f"status {value!r}",
            f"status {value + chr(34)!r}",
            str([value, "kept"]),
            str({"field": value}),
        ):
            scrubbed = redactor.scrub(text)
            assert value not in _literals(scrubbed)
            assert value not in scrubbed
    assert redactor.scrub(str(["kept", 1])) == "['kept', 1]"


def test_line_scrubbing_removes_a_complete_multiline_value() -> None:
    # No line of it holds a letter or digit, so only the whole value can match.
    secret = "!@#$%^&*()[]{}<>?\n?<>}{][)(*&^%$#@!"
    redactor = SecretRedactor([secret])
    text = f"upload rejected\n{secret}\n{{\n]\nRetry with: osmosis eval upload run-1"
    scrubbed = redactor.scrub_lines(text)
    assert secret not in scrubbed
    assert scrubbed.splitlines() == [
        "upload rejected",
        "[REDACTED]",
        "{",
        "]",
        "Retry with: osmosis eval upload run-1",
    ]
    # Its lines on their own are formatting, as before.
    first, second = secret.split("\n")
    assert redactor.scrub_lines(f"{first}\nkept\n{second}") == (
        f"{first}\nkept\n{second}"
    )


def test_data_too_deep_to_walk_keeps_its_shape_with_a_marker() -> None:
    # Far deeper than the recursion limit, and still accepted by the parser.
    depth = 5000
    data = json.loads('{"count": 2, "deep": ' + "[" * depth + "]" * depth + "}")
    result = SecretRedactor(["synth-secret-key"]).scrub_value(data)
    assert result["count"] == 2
    node, levels = result["deep"], 0
    while isinstance(node, list):
        node, levels = node[0], levels + 1
    assert node == "[REDACTED]"
    assert levels < depth
    json.dumps(result)
