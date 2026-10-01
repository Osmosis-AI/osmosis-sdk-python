"""Native evidence redaction patterns stay linear without changing redactions."""

from __future__ import annotations

import random
import re
import signal

from osmosis_ai.rollout.backend.harbor import evidence

# The patterns shipped before the linear rewrite, kept here so the rewrite is
# proven to redact exactly the same text rather than approximately.
LEGACY_ASSIGNMENT = re.compile(
    r"""(?ix)([\w-]*(?:api[_-]?key|authorization|password|credential|secret|token)[\w-]*\s*[=:]\s*)("[^"\n]*"|'[^'\n]*'|[^\s,;\]}]+)"""
)
LEGACY_URL_USERINFO = re.compile(
    r"([a-z][a-z0-9+.-]*://)[^\s/@]+:[^\s/@]+@", re.IGNORECASE
)


def assignment(pattern, text):
    return pattern.sub(lambda match: match[1] + "[REDACTED]", text)


def userinfo(pattern, text):
    return pattern.sub(r"\1[REDACTED]@", text)


def test_redaction_matches_the_legacy_patterns_on_edge_cases():
    cases = [
        (
            'API_KEY=abc PASSWORD="private value" SECRET_TOKEN : xyz',
            "API_KEY=[REDACTED] PASSWORD=[REDACTED] SECRET_TOKEN : [REDACTED]",
            None,
        ),
        (
            "not_a_password_hint: foo token=123; authorization=Bearer",
            "not_a_password_hint: [REDACTED] token=[REDACTED]; authorization=[REDACTED]",
            None,
        ),
        (
            "http://username:password@host/path 1http://u:p@host",
            None,
            "http://[REDACTED]@host/path 1http://[REDACTED]@host",
        ),
        (
            "111://u:p@host h://:p@host h://u:@host h://::@host",
            None,
            "111://u:p@host h://:p@host h://u:@host h://::@host",
        ),
        (
            "h://a:b:c@host @ h://a:b/@host h://:@host http://a:b@c:d@e",
            None,
            "h://[REDACTED]@host @ h://a:b/@host h://:@host http://[REDACTED]@c:d@e",
        ),
        (
            "+1.x-http://u:p@h a.b-c+d://user:pw@h a://b://c:d@e",
            None,
            "+1.x-http://[REDACTED]@h a.b-c+d://[REDACTED]@h a://b://[REDACTED]@e",
        ),
        (
            'api_key="escaped\\" value" token\n=\n123',
            'api_key=[REDACTED] value" token\n=\n[REDACTED]',
            None,
        ),
        (
            "tokenfoo=bar 123token=bar --token=bar token-token : z",
            "tokenfoo=[REDACTED] 123token=[REDACTED] --token=[REDACTED] token-token : [REDACTED]",
            None,
        ),
        (
            "token=\"a\"password=b token='a'x-token=1 token=a,token=b;token=c]t=d}",
            "token=[REDACTED]password=[REDACTED] token=[REDACTED]x-token=[REDACTED] token=[REDACTED],token=[REDACTED];token=[REDACTED]]t=d}",
            None,
        ),
        (
            "X-API-KEY:abc x-api-key: 'q r' passwordless=1 my_secret_value = 'q'",
            "X-API-KEY:[REDACTED] x-api-key: [REDACTED] passwordless=[REDACTED] my_secret_value = [REDACTED]",
            None,
        ),
        # Unicode case folding: long s and the Kelvin sign match s and k.
        (
            "\u017fecret=1 to\u212aen=2 \u0130token=3 h\u0130://u:p@h",
            "\u017fecret=[REDACTED] to\u212aen=[REDACTED] \u0130token=[REDACTED] h\u0130://u:p@h",
            "\u017fecret=1 to\u212aen=2 \u0130token=3 h\u0130://[REDACTED]@h",
        ),
    ]
    for text, expected_assignment, expected_userinfo in cases:
        actual_assignment = assignment(evidence._ASSIGNMENT, text)
        actual_userinfo = userinfo(evidence._URL_USERINFO, text)
        if expected_assignment is None:
            assert actual_assignment == text
        else:
            assert actual_assignment == expected_assignment
        assert actual_userinfo == (
            expected_userinfo if expected_userinfo is not None else text
        )


def test_patterns_redact_exactly_what_the_legacy_patterns_redact():
    chunks = ["token", "foo", "api-key", "api_key", "apikey", "password", "user"]
    chunks += ["AUTHORIZATION", "credential", "secret", "http", "://", "123", "pass"]
    chunks += ["x", "-", ".", "+", "_", ":", "/", "=", "@", ",", ";", "]", "}"]
    chunks += [" ", "\n", "\t", '"', "'", "\u0130", "\u017f", "\u212a"]
    rng = random.Random(119)
    examples = [
        "".join(rng.choices(chunks, k=rng.randrange(1, 40))) for _ in range(20000)
    ]
    for text in examples:
        assert assignment(evidence._ASSIGNMENT, text) == assignment(
            LEGACY_ASSIGNMENT, text
        )
        assert userinfo(evidence._URL_USERINFO, text) == userinfo(
            LEGACY_URL_USERINFO, text
        )


def test_patterns_stay_linear_on_long_runs():
    # The legacy patterns take seconds on 16 KB of any of these and hours on 1 MB.
    megabyte = 1_000_000
    inputs = [
        "A" * megabyte,
        "a1b2c3d4" * (megabyte // 8),
        "-" * megabyte,
        "token" * (megabyte // 5),
        "api_key=" + "x" * megabyte,
        "123" * (megabyte // 3),
        "http://" + ":" * megabyte,
        "http://" + "a:" * (megabyte // 2),
    ]

    def expire(signum, frame):
        raise TimeoutError("redaction backtracks on long runs")

    previous = signal.signal(signal.SIGALRM, expire)
    signal.alarm(10)
    try:
        for text in inputs:
            assignment(evidence._ASSIGNMENT, text)
            userinfo(evidence._URL_USERINFO, text)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)
