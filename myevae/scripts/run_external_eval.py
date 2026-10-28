#!/usr/bin/env python3
"""
External cohort inference (zero tuning on external data).

Example:
  python scripts/run_external_eval.py \\
    --model-dir output/main_ablation/A1_s42 \\
    --config configs/external_eval.yaml \\
    --variant A1 --model-name MO-RiskVAE --seed 42 \\
    --output results/external_eval_raw.csv --append
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from bootstrap_ci import bootstrap_ci
from exp_utils import (
    compute_cindex,
    count_events,
    find_checkpoint,
    load_myevae_model_and_dataset,
    load_yaml,
    predict_risk_myevae,
    repo_root,
)

EXTERNAL_COLUMNS = [
    "model",
    "variant",
    "seed",
    "cohort",
    "n",
    "events",
    "available_modalities",
    "feature_overlap",
    "missing_handling",
    "cindex",
    "ci_low",
    "ci_high",
    "checkpoint_path",
]


def load_cohort_df(
    feature_path: Path,
    label_path: Path,
    event_col: str,
    time_col: str,
) -> pd.DataFrame:
    feat = pd.read_csv(feature_path, index_col=0)
    lab = pd.read_csv(label_path, index_col=0)[[event_col, time_col]]
    return pd.concat([lab, feat], axis=1)


def load_modality_info(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def bootstrap_cindex_ci(
    event: np.ndarray,
    time: np.ndarray,
    risk: np.ndarray,
    *,
    event_is_censored: bool,
    n_bootstrap: int,
    ci_level: float,
    seed: int,
) -> tuple[float, float, float]:
    """Patient-level bootstrap CI for C-index."""
    n = len(time)
    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            c = compute_cindex(
                event[idx], time[idx], risk[idx], event_is_censored=event_is_censored
            )
            scores.append(c)
        except Exception:
            continue
    if not scores:
        c = compute_cindex(event, time, risk, event_is_censored=event_is_censored)
        return c, c, c
    scores = np.sort(scores)
    alpha = (1.0 - ci_level) / 2.0
    return (
        compute_cindex(event, time, risk, event_is_censored=event_is_censored),
        float(np.quantile(scores, alpha)),
        float(np.quantile(scores, 1.0 - alpha)),
    )


def append_external_row(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row], columns=EXTERNAL_COLUMNS)
    write_header = not path.exists()
    df.to_csv(path, mode="a", header=write_header, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="External cohort evaluation")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--cohort", type=str, default=None, help="Run single cohort only")
    parser.add_argument("--src-dir", type=Path, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Skip model; check paths only")
    args = parser.parse_args()

    cfg_path = args.config or (repo_root() / "configs" / "external_eval.yaml")
    cfg = load_yaml(cfg_path)
    external_root = Path(cfg["paths"]["external_root"])
    data_cfg = cfg["data"]
    event_col = data_cfg["event_col"]
    time_col = data_cfg["time_col"]
    event_is_censored = bool(data_cfg.get("event_is_censored", False))
    eval_cfg = cfg.get("eval", {})
    n_boot = int(eval_cfg.get("bootstrap_n", 1000))
    ci_level = float(eval_cfg.get("ci_level", 0.95))
    ckpt_glob = eval_cfg.get("checkpoint_glob", "test_model_os.pth")

    if not args.append and args.output.exists():
        args.output.unlink()

    if args.dry_run:
        for cohort in cfg.get("cohorts", []):
            name = cohort["name"]
            if args.cohort and args.cohort != name:
                continue
            feat_path = external_root / cohort["feature_path"]
            lab_path = external_root / cohort["label_path"]
            ok = feat_path.exists() and lab_path.exists()
            print(f"[DRY-RUN] {name}: {'OK' if ok else 'MISSING'} {feat_path}")
        return

    ckpt = find_checkpoint(args.model_dir, ckpt_glob)
    src = args.src_dir or (repo_root() / "src")

    train_cfg = {
        "data": data_cfg,
        "training": {"z_dim": 32, "layer_dims": [[32, 4], [16, 4], [4, 1], [4, 1], [4, 1], [4, 1]]},
    }

    for cohort in cfg.get("cohorts", []):
        name = cohort["name"]
        if args.cohort and args.cohort != name:
            continue

        feat_path = external_root / cohort["feature_path"]
        lab_path = external_root / cohort["label_path"]
        mod_path = external_root / cohort.get("modality_info", "")
        if not mod_path.exists():
            mod_path = None

        if not feat_path.exists() or not lab_path.exists():
            print(f"[SKIP] {name}: missing {feat_path} or {lab_path}")
            continue

        mod_info = load_modality_info(mod_path if mod_path else None)
        available = cohort.get("available_modalities") or mod_info.get(
            "available_modalities", "unknown"
        )
        overlap = cohort.get("feature_overlap") or mod_info.get("feature_overlap", "TBD")
        missing_h = cohort.get("missing_handling") or cfg["data"].get(
            "default_missing_handling", "unknown"
        )

        df = load_cohort_df(feat_path, lab_path, event_col, time_col)
        n, events = count_events(df, event_col, event_is_censored=event_is_censored)

        model, dataset = load_myevae_model_and_dataset(
            ckpt, df, train_cfg, src_dir=src, K=args.K
        )
        risk = predict_risk_myevae(model, dataset)
        event = df[event_col].values
        time = df[time_col].values.astype(float)

        cindex, ci_lo, ci_hi = bootstrap_cindex_ci(
            event,
            time,
            risk,
            event_is_censored=event_is_censored,
            n_bootstrap=n_boot,
            ci_level=ci_level,
            seed=args.seed,
        )

        row = {
            "model": args.model_name,
            "variant": args.variant,
            "seed": args.seed,
            "cohort": name,
            "n": n,
            "events": events,
            "available_modalities": available,
            "feature_overlap": overlap,
            "missing_handling": missing_h,
            "cindex": round(cindex, 6),
            "ci_low": round(ci_lo, 6),
            "ci_high": round(ci_hi, 6),
            "checkpoint_path": str(ckpt.resolve()),
        }
        append_external_row(args.output, row)
        print(f"{args.model_name} {name} C-index={cindex:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

    print(f"[OK] Results → {args.output}")


if __name__ == "__main__":
    main()
