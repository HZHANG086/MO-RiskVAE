#!/usr/bin/env python3
"""Bootstrap CI for seed-level C-index values."""
from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np


def bootstrap_ci(
    values: Iterable[float],
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("No values for bootstrap")
    if arr.size == 1:
        v = float(arr[0])
        return v, v, v

    rng = np.random.default_rng(seed)
    n = arr.size
    means = []
    for _ in range(n_bootstrap):
        sample = arr[rng.integers(0, n, size=n)]
        means.append(sample.mean())
    means = np.sort(means)
    alpha = (1.0 - ci_level) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return float(arr.mean()), lo, hi


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap CI for a list of C-index values")
    parser.add_argument("--values", nargs="+", type=float, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    mean, lo, hi = bootstrap_ci(
        args.values, n_bootstrap=args.n_bootstrap, ci_level=args.ci_level, seed=args.seed
    )
    print(f"mean={mean:.6f}")
    print(f"ci_low={lo:.6f}")
    print(f"ci_high={hi:.6f}")


if __name__ == "__main__":
    main()
