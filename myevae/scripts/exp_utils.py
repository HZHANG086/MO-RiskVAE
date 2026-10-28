"""Shared utilities for MO-RiskVAE experiment scripts (no model changes)."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml

# Default server layout; override with MYEVAE_ROOT
DEFAULT_MYEVAE_ROOT = os.environ.get("MYEVAE_ROOT", "/work/zh/myevae")
DEFAULT_GUMBEL_ROOT = os.environ.get("GUMBEL_ROOT", "/work/zh/gumbel-softmax-vae")


def repo_root() -> Path:
    return Path(os.environ.get("MYEVAE_ROOT", DEFAULT_MYEVAE_ROOT)).resolve()


def gumbel_root() -> Path:
    return Path(os.environ.get("GUMBEL_ROOT", DEFAULT_GUMBEL_ROOT)).resolve()


def scripts_dir() -> Path:
    return Path(__file__).resolve().parent


def _expand_env(value: str) -> str:
    """Expand ${VAR:default} and $VAR in YAML strings."""

    def repl(match: re.Match) -> str:
        body = match.group(1)
        if ":" in body:
            var, default = body.split(":", 1)
            return os.environ.get(var, default)
        return os.environ.get(body, "")

    out = re.sub(r"\$\{([^}]+)\}", repl, value)
    return os.path.expandvars(out)


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return _expand_dict(raw)


def _expand_dict(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _expand_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_dict(v) for v in obj]
    if isinstance(obj, str):
        return _expand_env(obj)
    return obj


def resolve_path(base: Path, maybe_rel: str) -> Path:
    p = Path(_expand_env(maybe_rel))
    if p.is_absolute():
        return p
    return (base / p).resolve()


def ensure_src_on_path(src_dir: Optional[Path] = None) -> Path:
    src = src_dir or (repo_root() / "src")
    import sys

    s = str(src)
    if s not in sys.path:
        sys.path.insert(0, s)
    return src


def load_split_dataframe(
    inputs_dir: Path,
    split: str,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    data_cfg = cfg.get("data", cfg)
    event_col = data_cfg["event_col"]
    time_col = data_cfg["time_col"]
    labels_path = inputs_dir / data_cfg.get("labels", "labels.csv")

    if split == "train":
        feat_name = data_cfg.get("train_features", "train_features_os_processed.csv")
    elif split == "valid":
        feat_name = data_cfg.get("valid_features", "valid_features_os_processed.csv")
    else:
        raise ValueError(f"Unknown split: {split}")

    feat_path = inputs_dir / feat_name
    if not feat_path.exists():
        raise FileNotFoundError(f"Features not found: {feat_path}")

    feat = pd.read_csv(feat_path, index_col=0)
    lab = pd.read_csv(labels_path, index_col=0)[[event_col, time_col]]
    return pd.concat([lab, feat], axis=1)


def find_checkpoint(model_dir: Path, checkpoint_glob: str = "test_model_os.pth") -> Path:
    model_dir = Path(model_dir)
    direct = model_dir / checkpoint_glob
    if direct.exists():
        return direct
    matches = sorted(model_dir.rglob(checkpoint_glob))
    if not matches:
        raise FileNotFoundError(f"No checkpoint matching {checkpoint_glob} under {model_dir}")
    return matches[0]


def find_training_json(model_dir: Path, json_glob: str = "test_model_os.json") -> Optional[Path]:
    model_dir = Path(model_dir)
    direct = model_dir / json_glob
    if direct.exists():
        return direct
    matches = sorted(model_dir.rglob(json_glob))
    return matches[0] if matches else None


def event_indicator_array(df: pd.DataFrame, event_col: str, event_is_censored: bool = False) -> np.ndarray:
    """Boolean event indicator for sksurv (True = event occurred)."""
    v = df[event_col].astype(bool).values
    if event_is_censored:
        return ~v
    return v


def count_events(df: pd.DataFrame, event_col: str, event_is_censored: bool = False) -> Tuple[int, int]:
    n = len(df)
    events = int(event_indicator_array(df, event_col, event_is_censored).sum())
    return n, events


def compute_cindex(
    event: np.ndarray,
    time: np.ndarray,
    risk: np.ndarray,
    *,
    event_is_censored: bool = False,
) -> float:
    from sksurv.metrics import concordance_index_censored

    event_bool = event.astype(bool)
    if event_is_censored:
        event_bool = ~event_bool
    return float(concordance_index_censored(event_bool, time.astype(float), risk.astype(float))[0])


EVAL_COLUMNS = [
    "model",
    "variant",
    "seed",
    "split",
    "cindex",
    "n",
    "events",
    "checkpoint_path",
]


def append_csv_row(path: Path, row: Dict[str, Any], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row], columns=columns)
    header = not path.exists()
    df.to_csv(path, mode="a", header=header, index=False)


def load_myevae_model_and_dataset(
    checkpoint: Path,
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    src_dir: Optional[Path] = None,
    z_dim: int = 32,
    K: Optional[int] = None,
):
    """Load MultiModalVAE from checkpoint (requires myevae on PYTHONPATH)."""
    import torch
    from torch.nn import LeakyReLU, Tanh

    ensure_src_on_path(src_dir)
    from myevae.dataset import Dataset
    from myevae.nn import MultiModalVAE as Model

    data_cfg = cfg.get("data", cfg)
    input_types = list(data_cfg["input_types"])
    input_types_sub = list(data_cfg.get("input_types_subtask", ["clin"]))
    all_types = input_types + input_types_sub
    event_col = data_cfg["event_col"]
    time_col = data_cfg["time_col"]

    dataset = Dataset(
        df,
        all_types,
        event_indicator_col=event_col,
        event_time_col=time_col,
        device="cpu",
    )
    input_dims = [getattr(dataset, f"X_{t}").shape[1] for t in input_types]
    input_dims_sub = [getattr(dataset, f"X_{t}").shape[1] for t in input_types_sub]

    train_cfg = cfg.get("training", {})
    layer_dims = train_cfg.get(
        "layer_dims",
        [[32, 4], [16, 4], [4, 1], [4, 1], [4, 1], [4, 1]],
    )
    layer_dims_sub = train_cfg.get("layer_dims_subtask", [4, 1])

    kwargs = dict(
        input_types=input_types,
        input_dims=input_dims,
        layer_dims=layer_dims,
        input_types_subtask=input_types_sub,
        input_dims_subtask=input_dims_sub,
        layer_dims_subtask=layer_dims_sub,
        z_dim=train_cfg.get("z_dim", z_dim),
        activation=LeakyReLU(0.01),
        subtask_activation=Tanh(),
        dropout=train_cfg.get("dropout", 0.0),
        dropout_subtask=train_cfg.get("dropout_subtask", 0.0),
    )
    if K is not None:
        kwargs["K"] = K

    model = Model(**kwargs)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model, dataset


def build_modality_tensors_from_dataset(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    *,
    src_dir: Optional[Path] = None,
) -> Dict[str, torch.Tensor]:
    """Extract per-modality tensors via myevae.Dataset (fair feature layout)."""
    import torch
    from torch.utils.data import DataLoader

    ensure_src_on_path(src_dir)
    from myevae.dataset import Dataset

    data_cfg = cfg.get("data", cfg)
    input_types = list(data_cfg["input_types"])
    input_types_sub = list(data_cfg.get("input_types_subtask", ["clin"]))
    all_types = input_types + input_types_sub
    event_col = data_cfg["event_col"]
    time_col = data_cfg["time_col"]

    dataset = Dataset(
        df,
        all_types,
        event_indicator_col=event_col,
        event_time_col=time_col,
        device="cpu",
    )
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    batch = next(iter(loader))[1]

    tensors: Dict[str, torch.Tensor] = {}
    for t in input_types + input_types_sub:
        key = f"X_{t}"
        if key in batch:
            tensors[t] = batch[key].float()
    tensors["time"] = (
        batch[time_col].float()
        if time_col in batch
        else torch.tensor(df[time_col].values.astype(np.float32))
    )
    event_is_censored = bool(data_cfg.get("event_is_censored", False))
    if event_col in batch:
        ev = batch[event_col].bool()
    elif "event" in batch:
        ev = batch["event"].bool()
    else:
        ev = torch.tensor(df[event_col].astype(bool).values)
    tensors["event"] = (~ev if event_is_censored else ev).float()
    return tensors


def predict_risk_myevae(model, dataset, batch_size: int = 1024) -> np.ndarray:
    import torch
    from torch.utils.data import DataLoader

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    risks = []
    for _, data in enumerate(loader):
        with torch.no_grad():
            iv = [data[f"X_{t}"] for t in model.input_types_vae]
            it = [data[f"X_{t}"] for t in model.input_types_subtask]
            _, _, _, _, rp = model.forward((iv, it))
            risks.append(rp.flatten().numpy())
    return np.concatenate(risks)
