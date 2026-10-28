#!/usr/bin/env python3
"""Paired comparison of C-index between two variants (same seeds)."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats
except ImportError:
    stats = None

from bootstrap_ci import bootstrap_ci


def paired_comparison(
    df: pd.DataFrame,
    variant_a: str,
    variant_b: str,
    metric: str = "cindex",
    split: str = "valid",
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict:
    da = df[(df["variant"] == variant_a) & (df["split"] == split)].copy()
    db = df[(df["variant"] == variant_b) & (df["split"] == split)].copy()
    merged = da.merge(
        db,
        on="seed",
        suffixes=("_a", "_b"),
        how="inner",
    )
    if merged.empty:
        raise ValueError(
            f"No paired seeds for {variant_a} vs {variant_b} on split={split}"
        )

    a_vals = merged[f"{metric}_a"].astype(float).values
    b_vals = merged[f"{metric}_b"].astype(float).values
    diffs = a_vals - b_vals

    stat, p_wilcoxon = np.nan, np.nan
    if stats is not None:
        try:
            stat, p_wilcoxon = stats.wilcoxon(a_vals, b_vals, zero_method="wilcox")
        except ValueError:
            pass

    rng = np.random.default_rng(seed)
    boot_diffs = []
    n = len(diffs)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_diffs.append(float(np.mean(diffs[idx])))
    boot_diffs = np.sort(boot_diffs)
    ci_low = float(np.quantile(boot_diffs, 0.025))
    ci_high = float(np.quantile(boot_diffs, 0.975))

    return {
        "variant_a": variant_a,
        "variant_b": variant_b,
        "split": split,
        "n_pairs": n,
        "seeds": list(merged["seed"].astype(int)),
        "mean_a": float(np.mean(a_vals)),
        "mean_b": float(np.mean(b_vals)),
        "mean_diff": float(np.mean(diffs)),
        "wilcoxon_stat": float(stat) if stat == stat else None,
        "wilcoxon_p": float(p_wilcoxon) if p_wilcoxon == p_wilcoxon else None,
        "bootstrap_paired_ci_low": ci_low,
        "bootstrap_paired_ci_high": ci_high,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired test between two variants")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--a", required=True, help="Variant A (e.g. A1)")
    parser.add_argument("--b", required=True, help="Variant B (e.g. A0)")
    parser.add_argument("--metric", default="cindex")
    parser.add_argument("--split", default="valid")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    res = paired_comparison(
        df,
        variant_a=args.a,
        variant_b=args.b,
        metric=args.metric,
        split=args.split,
        n_bootstrap=args.n_bootstrap,
    )

    print(f"{res['variant_a']} vs {res['variant_b']} (split={res['split']}, n={res['n_pairs']})")
    print(f"seeds: {res['seeds']}")
    print(f"mean {args.a} = {res['mean_a']:.6f}")
    print(f"mean {args.b} = {res['mean_b']:.6f}")
    print(f"mean diff ({args.a} - {args.b}) = {res['mean_diff']:.6f}")
    print(f"Wilcoxon p = {res['wilcoxon_p']}")
    print(
        f"bootstrap paired 95% CI = [{res['bootstrap_paired_ci_low']:.6f}, "
        f"{res['bootstrap_paired_ci_high']:.6f}]"
    )


if __name__ == "__main__":
    main()
