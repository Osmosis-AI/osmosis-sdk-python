"""Tests for eval function loader behavior."""

from __future__ import annotations

import pytest

from osmosis_ai.rollout.eval.evaluation.eval_fn import EvalFnError, load_eval_fns


def _dummy_eval(solution_str: str, ground_truth: str, extra_info: dict) -> float:
    return 1.0


def test_load_eval_fns_uses_module_path_as_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "osmosis_ai.rollout.eval.evaluation.eval_fn.load_eval_fn",
        lambda _path: _dummy_eval,
    )

    wrappers = load_eval_fns(["my_eval_module:score"])

    assert len(wrappers) == 1
    assert wrappers[0].name == "my_eval_module:score"


def test_load_eval_fns_rejects_duplicate_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "osmosis_ai.rollout.eval.evaluation.eval_fn.load_eval_fn",
        lambda _path: _dummy_eval,
    )

    with pytest.raises(EvalFnError, match="Duplicate eval function path"):
        load_eval_fns(["my_eval_module:score", "my_eval_module:score"])
