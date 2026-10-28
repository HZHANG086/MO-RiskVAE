#!/usr/bin/env python3
"""Aggregate per-seed eval rows into mean/std/CI summary."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bootstrap_ci import bootstrap_ci


EVAL_COLUMNS = [
    "model",
    "variant",
    "seed",
    "split",
    "cindex",
    "n",
    "events"
    "checkpoint_path",
]

SUMMARY_COLUMNS = [
    "variant",
    "model",
    "split",
    "n_seeds",
    "mean_cindex",
    "std_cindex",
    "ci_low",
    "ci_high",
]


def aggregate(
    df: pd.DataFrame,
    split: str | None = None,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
) -> pd.DataFrame:
    if split:
        df = df[df["split"] == split].copy()
    if df.empty:
        raise ValueError("No rows after split filter")

    rows = []
    for (variant, model, spl), g in df.groupby(["variant", "model", "split"], dropna=False):
        vals = g["cindex"].astype(float).values
        mean, lo, hi = bootstrap_ci(vals, n_bootstrap=n_bootstrap, ci_level=ci_level)
        rows.append(
            {
                "variant": variant,
                "model": model,
                "split": spl,
                "n_seeds": len(vals),
                "mean_cindex": mean,
                "std_cindex": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate multi-seed C-index results")
    parser.add_argument("--input", type=Path, required=True, help="Raw eval CSV")
    parser.add_argument("--output", type=Path, required=True, help="Summary CSV")
    parser.add_argument("--split", type=str, default="valid", help="Filter split (default: valid)")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--ci-level", type=float, default=0.95)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    for col in ("variant", "cindex", "split"):
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {args.input}")

    summary = aggregate(
        df, split=args.split, n_bootstrap=args.n_bootstrap, ci_level=args.ci_level
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output, index=False)
    print(summary.to_string(index=False))
    print(f"\n[OK] Wrote {args.output}")


if __name__ == "__main__":
    main()
