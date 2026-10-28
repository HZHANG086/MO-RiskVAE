#!/usr/bin/env python3
"""
Encoder + Cox baseline (no VAE / recon / KL).
Fair to A1: same inputs, z_dim=32, Adam lr, early stopping on valid C-index.

Example:
  python scripts/run_encoder_cox.py --config configs/baselines.yaml --seed 42
  python scripts/run_encoder_cox.py --config configs/baselines.yaml --seed 42 --run-coxph
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from exp_utils import (
    EVAL_COLUMNS,
    append_csv_row,
    build_modality_tensors_from_dataset,
    compute_cindex,
    count_events,
    event_indicator_array,
    load_split_dataframe,
    load_yaml,
    repo_root,
)


class ModalityEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: List[int], out_dim: int, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.LeakyReLU(0.01)])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderCoxModel(nn.Module):
    """Per-modality MLP encoders → concat → linear risk (log hazard)."""

    def __init__(
        self,
        input_dims: Dict[str, int],
        layer_dims: List[List[int]],
        input_types_sub: List[str],
        layer_dims_sub: List[int],
        z_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_types = list(input_dims.keys())
        self.input_types_sub = input_types_sub
        encoders = {}
        for i, t in enumerate(self.input_types):
            h = layer_dims[i] if i < len(layer_dims) else [32, 4]
            encoders[t] = ModalityEncoder(input_dims[t], h, z_dim // max(len(self.input_types), 1), dropout)
        self.encoders = nn.ModuleDict(encoders)
        sub_in = sum(input_dims[t] for t in input_types_sub)
        self.sub_encoder = ModalityEncoder(sub_in, layer_dims_sub, z_dim // 4, dropout) if sub_in else None
        fused_dim = sum(z_dim // max(len(self.input_types), 1) for _ in self.input_types)
        if self.sub_encoder:
            fused_dim += z_dim // 4
        self.risk_head = nn.Linear(fused_dim, 1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts = []
        for t in self.input_types:
            parts.append(self.encoders[t](batch[t]))
        if self.sub_encoder is not None:
            sub_x = torch.cat([batch[t] for t in self.input_types_sub], dim=1)
            parts.append(self.sub_encoder(sub_x))
        h = torch.cat(parts, dim=1)
        return self.risk_head(h).squeeze(-1)


def cox_partial_likelihood_loss(risk: torch.Tensor, time: torch.Tensor, event: torch.Tensor) -> torch.Tensor:
    """Negative log partial likelihood (Breslow). event: 1=event, 0=censored."""
    order = torch.argsort(time, descending=True)
    risk = risk[order]
    time = time[order]
    event = event[order]
    exp_risk = torch.exp(risk)
    log_cumsum = torch.log(torch.cumsum(exp_risk, dim=0) + 1e-8)
    loss = -(risk - log_cumsum) * event
    denom = event.sum().clamp(min=1.0)
    return loss.sum() / denom


def train_encoder_cox(
    model: EncoderCoxModel,
    train_tensors: Dict[str, torch.Tensor],
    valid_tensors: Dict[str, torch.Tensor],
    types: List[str],
    sub_types: List[str],
    cfg: Dict,
) -> Tuple[EncoderCoxModel, float]:
    enc_cfg = cfg["encoder_cox"]
    lr = enc_cfg["lr"]
    epochs = enc_cfg["epochs"]
    burn_in = enc_cfg["burn_in"]
    patience = enc_cfg["patience"]
    batch_size = enc_cfg["batch_size"]
    device = torch.device(enc_cfg.get("device", "cpu"))

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_cindex = -1.0
    best_state = None
    wait = 0

    def run_epoch(tensors: Dict[str, torch.Tensor], train: bool) -> float:
        n = len(tensors["time"])
        idx = np.arange(n)
        if train:
            np.random.shuffle(idx)
        total_loss = 0.0
        steps = 0
        model.train(train)
        for start in range(0, n, batch_size):
            sl = idx[start : start + batch_size]
            batch = {k: tensors[k][sl].to(device) for k in tensors if k not in ("time", "event")}
            time_b = tensors["time"][sl].to(device)
            event_b = tensors["event"][sl].to(device)
            risk = model(batch)
            loss = cox_partial_likelihood_loss(risk, time_b, event_b)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item()
            steps += 1
        return total_loss / max(steps, 1)

    def eval_cindex(tensors: Dict[str, torch.Tensor]) -> float:
        model.eval()
        with torch.no_grad():
            batch = {k: tensors[k].to(device) for k in tensors if k not in ("time", "event")}
            risk = model(batch).cpu().numpy()
        ev = tensors["event"].numpy().astype(bool)
        tm = tensors["time"].numpy()
        from sksurv.metrics import concordance_index_censored

        return float(concordance_index_censored(ev, tm, risk)[0])

    for epoch in range(epochs):
        run_epoch(train_tensors, train=True)
        if epoch < burn_in:
            continue
        vc = eval_cindex(valid_tensors)
        if vc > best_cindex:
            best_cindex = vc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_cindex


def concat_features(df: pd.DataFrame, types: List[str]) -> np.ndarray:
    blocks = []
    for t in types:
        cols = [c for c in df.columns if c.startswith(f"{t}_")]
        if cols:
            blocks.append(df[cols].values)
    if not blocks:
        raise ValueError("No feature columns found for CoxPH")
    return np.hstack(blocks)


def run_coxph_baseline(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    types: List[str],
    sub_types: List[str],
    event_col: str,
    time_col: str,
    cfg: Dict,
) -> float:
    from sksurv.linear_model import CoxnetSurvivalAnalysis
    from sksurv.util import Surv

    all_types = types + sub_types
    X_train = concat_features(train_df, all_types)
    X_valid = concat_features(valid_df, all_types)
    cox_cfg = cfg["coxph"]
    event_is_censored = cfg.get("data", {}).get("event_is_censored", False)
    ev_train = event_indicator_array(train_df, event_col, event_is_censored)
    y_train = Surv.from_arrays(event=ev_train, time=train_df[time_col].values.astype(float))
    model = CoxnetSurvivalAnalysis(
        l1_ratio=cox_cfg.get("l1_ratio", 0.5),
        alphas=[cox_cfg.get("alpha", 0.01)],
        max_iter=cox_cfg.get("max_iter", 10000),
    )
    model.fit(X_train, y_train)
    risk = model.predict(X_valid)
    event = valid_df[event_col].values
    time = valid_df[time_col].values.astype(float)
    return compute_cindex(event, time, risk, event_is_censored=event_is_censored)


def modality_dims(df: pd.DataFrame, types: List[str]) -> Dict[str, int]:
    dims = {}
    for t in types:
        cols = [c for c in df.columns if c.startswith(f"{t}_")]
        dims[t] = len(cols) if cols else 1
    return dims


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Encoder+Cox or CoxPH baseline")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-coxph", action="store_true", help="Run ElasticNet CoxPH instead of Encoder+Cox")
    parser.add_argument("--output", type=Path, default=None, help="Append eval row to this CSV")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg_path = args.config or (repo_root() / "configs" / "baselines.yaml")
    cfg = load_yaml(cfg_path)
    inputs_dir = Path(cfg["paths"]["inputs_dir"])
    out_root = Path(cfg["paths"]["output_root"])
    results_dir = Path(cfg["paths"]["results_dir"])

    train_df = load_split_dataframe(inputs_dir, "train", cfg)
    valid_df = load_split_dataframe(inputs_dir, "valid", cfg)
    data_cfg = cfg["data"]
    types = list(data_cfg["input_types"])
    sub_types = list(data_cfg.get("input_types_subtask", []))
    event_col = data_cfg["event_col"]
    time_col = data_cfg["time_col"]
    event_is_censored = bool(data_cfg.get("event_is_censored", False))

    if args.run_coxph:
        variant = cfg["coxph"]["variant"]
        model_name = cfg["coxph"]["model_name"]
        cindex = run_coxph_baseline(train_df, valid_df, types, sub_types, event_col, time_col, cfg)
        out_dir = out_root / f"COXPH_s{args.seed}"
    else:
        enc_cfg = cfg["encoder_cox"]
        variant = enc_cfg["variant"]
        model_name = enc_cfg["model_name"]
        out_dir = out_root / f"ENC_s{args.seed}"
        out_dir.mkdir(parents=True, exist_ok=True)

        src = repo_root() / "src"
        train_t = build_modality_tensors_from_dataset(train_df, cfg, src_dir=src)
        valid_t = build_modality_tensors_from_dataset(valid_df, cfg, src_dir=src)

        dims = {t: train_t[t].shape[1] for t in types}
        for t in sub_types:
            dims[t] = train_t[t].shape[1]

        model = EncoderCoxModel(
            input_dims={k: dims[k] for k in types},
            layer_dims=enc_cfg["layer_dims"],
            input_types_sub=sub_types,
            layer_dims_sub=enc_cfg["layer_dims_subtask"],
            z_dim=enc_cfg["z_dim"],
            dropout=enc_cfg.get("dropout", 0.0),
        )
        model, cindex = train_encoder_cox(model, train_t, valid_t, types, sub_types, cfg)
        ckpt = out_dir / "encoder_cox.pth"
        torch.save(model.state_dict(), ckpt)
        meta = {"seed": args.seed, "valid_cindex": cindex, "variant": variant}
        (out_dir / "encoder_cox.json").write_text(json.dumps(meta, indent=2))

    n, events = count_events(valid_df, event_col, event_is_censored=event_is_censored)
    out_csv = args.output or (results_dir / cfg["eval"]["raw_csv"])
    row = {
        "model": model_name,
        "variant": variant,
        "seed": args.seed,
        "split": "valid",
        "cindex": round(cindex, 6),
        "n": n,
        "events": events,
        "checkpoint_path": str(out_dir),
    }
    append_csv_row(out_csv, row, EVAL_COLUMNS)
    print(f"{model_name} seed={args.seed} valid C-index={cindex:.4f}")
    print(f"[OK] Appended {out_csv}")


if __name__ == "__main__":
    main()
