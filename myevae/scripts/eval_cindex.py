#!/usr/bin/env python3
"""
Evaluate OS C-index for a trained MyeVAE checkpoint on train/valid split.

Example:
  python scripts/eval_cindex.py \\
    --model-dir output/main_ablation/A1_s42 \\
    --split valid --variant A1 --model-name MO-RiskVAE --seed 42 \\
    --output results/main_ablation_raw.csv --append
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from exp_utils import (
    EVAL_COLUMNS,
    append_csv_row,
    compute_cindex,
    count_events,
    find_checkpoint,
    load_myevae_model_and_dataset,
    load_split_dataframe,
    load_yaml,
    predict_risk_myevae,
    repo_root,
)


def build_cfg(args: argparse.Namespace) -> dict:
    if args.config:
        cfg = load_yaml(Path(args.config))
    else:
        default_cfg = repo_root() / "configs" / "ablation_main.yaml"
        cfg = load_yaml(default_cfg) if default_cfg.exists() else {"data": {}}
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate C-index for one checkpoint")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--split", choices=["train", "valid"], default="valid")
    parser.add_argument("--variant", required=True, help="e.g. A0, A1, A4")
    parser.add_argument("--model-name", required=True, help="e.g. MyeVAE, MO-RiskVAE")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--inputs-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-glob", default="test_model_os.pth")
    parser.add_argument("--src-dir", type=Path, default=None, help="myevae src (default: MYEVAE_ROOT/src)")
    parser.add_argument("--K", type=int, default=None, help="Gumbel K if applicable")
    args = parser.parse_args()

    cfg = build_cfg(args)
    root = repo_root()
    inputs_dir = args.inputs_dir or Path(cfg.get("paths", {}).get("inputs_dir", root / "inputs"))

    df = load_split_dataframe(inputs_dir, args.split, cfg)
    data_cfg = cfg.get("data", cfg)
    event_col = data_cfg["event_col"]
    time_col = data_cfg["time_col"]

    ckpt = find_checkpoint(args.model_dir, args.checkpoint_glob)
    src = args.src_dir or (root / "src")

    model, dataset = load_myevae_model_and_dataset(
        ckpt, df, cfg, src_dir=src, K=args.K
    )
    risk = predict_risk_myevae(model, dataset)

    # MyeVAE censos: True = censored
    event = df[event_col].values
    time = df[time_col].values.astype(float)
    event_is_censored = bool(data_cfg.get("event_is_censored", False))
    cindex = compute_cindex(event, time, risk, event_is_censored=event_is_censored)
    n, events = count_events(df, event_col, event_is_censored=event_is_censored)

    row = {
        "model": args.model_name,
        "variant": args.variant,
        "seed": args.seed,
        "split": args.split,
        "cindex": round(cindex, 6),
        "n": n,
        "events": events,
        "checkpoint_path": str(ckpt.resolve()),
    }

    out = args.output
    if not args.append and out.exists():
        out.unlink()
    append_csv_row(out, row, EVAL_COLUMNS)

    print(
        f"{args.model_name} {args.variant} seed={args.seed} split={args.split} "
        f"C-index={cindex:.4f} n={n} events={events}"
    )
    print(f"[OK] {'Appended' if args.append else 'Wrote'} {out}")


if __name__ == "__main__":
    main()
