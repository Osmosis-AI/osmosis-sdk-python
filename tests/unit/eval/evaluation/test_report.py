"""Tests for evaluation report utilities."""

from __future__ import annotations

from osmosis_ai.eval.evaluation.report import pass_at_k


def test_pass_at_k_all_correct() -> None:
    """When all samples are correct, pass@k should be 1.0."""
    assert pass_at_k(n=5, c=5, k=1) == 1.0


def test_pass_at_k_none_correct() -> None:
    """When no samples are correct, pass@k should be 0.0."""
    assert pass_at_k(n=5, c=0, k=1) == 0.0


def test_pass_at_k_n_leq_k() -> None:
    """When n <= k, pass@k should be 1.0 if any sample is correct."""
    assert pass_at_k(n=3, c=1, k=3) == 1.0
    assert pass_at_k(n=3, c=1, k=5) == 1.0


def test_pass_at_k_partial() -> None:
    """Partial correctness should give a value between 0 and 1."""
    result = pass_at_k(n=10, c=3, k=1)
    assert 0.0 < result < 1.0


def test_pass_at_k_high_k_few_failures() -> None:
    """When k > n - c (more picks than failures), pass@k should be 1.0."""
    # n=10, c=8, k=3 => n-c=2 < k=3, so pass@k = 1.0
    assert pass_at_k(n=10, c=8, k=3) == 1.0


def test_pass_at_k_exact_values() -> None:
    """Verify exact computation for a known case.

    n=4, c=2, k=2: 1 - C(2,2)/C(4,2) = 1 - 1/6 = 5/6
    """
    result = pass_at_k(n=4, c=2, k=2)
    assert abs(result - 5 / 6) < 1e-10
