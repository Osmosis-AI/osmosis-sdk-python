"""Evaluation reporting: pass@k estimator and utility functions."""

from __future__ import annotations

from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator.

    Computes the probability that at least one of k randomly selected
    samples from n total samples (of which c are correct) is correct.

    Formula: 1 - comb(n-c, k) / comb(n, k)
    """
    if n < 1 or k < 1:
        raise ValueError(f"pass_at_k: n and k must be >= 1, got n={n}, k={k}")
    if c < 0:
        raise ValueError(f"pass_at_k: c must be >= 0, got c={c}")
    if c > n:
        raise ValueError(f"pass_at_k: c ({c}) cannot exceed n ({n})")
    if c == 0:
        return 0.0
    if n <= k or n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


__all__ = ["pass_at_k"]
