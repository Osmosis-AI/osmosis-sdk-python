"""Tests for the shared interactive prompt key bindings."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator

import pytest
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.output import DummyOutput

from osmosis_ai.cli.prompts import confirm, pause, select_list, text_input

ESCAPE = "\x1b"
OPTION_LEFT = "\x1b[1;3D"
OPTION_RIGHT = "\x1b[1;3C"
OPTION_B = "\x1bb"
ENTER = "\r"

# A prompt that never answers hangs the suite, so bound every ask.
ASK_TIMEOUT = 10.0


@pytest.fixture
def keys() -> Iterator[PipeInput]:
    with create_pipe_input() as pipe_input:
        with create_app_session(input=pipe_input, output=DummyOutput()):
            yield pipe_input


async def _ask_text(
    pipe_input: PipeInput, typed: str, *, default: str = ""
) -> str | None:
    pipe_input.send_text(typed)
    return await asyncio.wait_for(
        asyncio.to_thread(text_input, "Where?", default=default),
        timeout=ASK_TIMEOUT,
    )


# ── Option/alt chords must edit, not cancel ──────────────────────────


async def test_option_left_jumps_a_word_back(keys: PipeInput) -> None:
    answer = await _ask_text(keys, f"one two{OPTION_LEFT}X{ENTER}")
    assert answer == "one Xtwo"


async def test_option_right_jumps_a_word_forward(keys: PipeInput) -> None:
    answer = await _ask_text(
        keys, f"{OPTION_LEFT}{OPTION_RIGHT}X{ENTER}", default="one two"
    )
    assert answer == "one twoX"


async def test_option_b_jumps_a_word_back(keys: PipeInput) -> None:
    answer = await _ask_text(keys, f"one two{OPTION_B}X{ENTER}")
    assert answer == "one Xtwo"


async def test_unbound_option_chord_is_ignored(keys: PipeInput) -> None:
    answer = await _ask_text(keys, f"path{ESCAPE}j{ENTER}")
    assert answer == "path"


async def test_option_left_does_not_cancel_a_select(keys: PipeInput) -> None:
    keys.send_text(f"{OPTION_LEFT}{ENTER}")
    choice = await asyncio.wait_for(
        asyncio.to_thread(select_list, "Pick?", ["https", "ssh"]),
        timeout=ASK_TIMEOUT,
    )
    assert choice == "https"


async def test_option_left_does_not_cancel_a_confirm(keys: PipeInput) -> None:
    keys.send_text(f"{OPTION_LEFT}y")
    answered = await asyncio.wait_for(
        asyncio.to_thread(confirm, "Proceed?"),
        timeout=ASK_TIMEOUT,
    )
    assert answered is True


# ── ESC on its own still cancels ─────────────────────────────────────


async def test_escape_cancels_text_input(keys: PipeInput) -> None:
    assert await _ask_text(keys, ESCAPE) is None


async def test_escape_cancels_select(keys: PipeInput) -> None:
    keys.send_text(ESCAPE)
    choice = await asyncio.wait_for(
        asyncio.to_thread(select_list, "Pick?", ["https", "ssh"]),
        timeout=ASK_TIMEOUT,
    )
    assert choice is None


async def test_pause_continues_on_enter(keys: PipeInput) -> None:
    keys.send_text(ENTER)
    continued = await asyncio.wait_for(
        asyncio.to_thread(pause, "Press Enter to continue"),
        timeout=ASK_TIMEOUT,
    )
    assert continued is True


async def test_pause_reports_a_cancel(keys: PipeInput) -> None:
    keys.send_text(ESCAPE)
    continued = await asyncio.wait_for(
        asyncio.to_thread(pause, "Press Enter to continue"),
        timeout=ASK_TIMEOUT,
    )
    assert continued is False


async def test_escape_cancels_confirm(keys: PipeInput) -> None:
    keys.send_text(ESCAPE)
    answered = await asyncio.wait_for(
        asyncio.to_thread(confirm, "Proceed?"),
        timeout=ASK_TIMEOUT,
    )
    assert answered is None
