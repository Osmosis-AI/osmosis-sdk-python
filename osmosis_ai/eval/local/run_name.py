"""Cloud-style names for local evaluation runs."""

from __future__ import annotations

import secrets

from osmosis_ai.eval.local.state import validate_run_name

_ADJECTIVES = (
    "brave",
    "calm",
    "daring",
    "eager",
    "fierce",
    "gentle",
    "happy",
    "keen",
    "lively",
    "mellow",
    "noble",
    "patient",
    "quick",
    "serene",
    "swift",
    "tender",
    "vivid",
    "warm",
    "zealous",
    "agile",
    "bold",
    "clever",
    "diligent",
    "elegant",
    "fluent",
    "graceful",
    "humble",
    "jolly",
    "kindly",
    "lucid",
    "mighty",
    "nimble",
    "polite",
    "radiant",
    "silent",
    "tranquil",
    "upbeat",
    "witty",
    "youthful",
    "zen",
)
_ANIMALS = (
    "falcon",
    "tiger",
    "dolphin",
    "eagle",
    "panther",
    "wolf",
    "hawk",
    "lion",
    "bear",
    "fox",
    "owl",
    "raven",
    "shark",
    "whale",
    "cobra",
    "jaguar",
    "lynx",
    "otter",
    "puma",
    "seal",
    "badger",
    "crane",
    "dove",
    "finch",
    "gecko",
    "heron",
    "ibis",
    "koala",
    "lemur",
    "manta",
    "newt",
    "oriole",
    "parrot",
    "quail",
    "robin",
    "stork",
    "toucan",
    "viper",
    "wombat",
    "zebra",
)


def generate_run_name() -> str:
    """Return a memorable ``adjective-animal-number`` name like cloud eval."""
    return validate_run_name(
        f"{secrets.choice(_ADJECTIVES)}-"
        f"{secrets.choice(_ANIMALS)}-"
        f"{secrets.randbelow(100)}"
    )
