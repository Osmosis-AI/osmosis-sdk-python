"""Evaluation reporting: pass@k estimator and utility functions."""

from __future__ import annotations

from math import comb


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator.

    Computes the probability that at least one of k randomly selected
    samples from n total samples (of which c are correct) is correct.

    Formula: 1 - comb(n-c, k) / comb(n, k)
    """
    if c == 0:
        return 0.0
    if n <= k:
        return 1.0
    if c >= n:
        return 1.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


__all__ = ["pass_at_k"]
