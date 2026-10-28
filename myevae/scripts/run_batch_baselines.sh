#!/usr/bin/env bash
# Batch 1 P0-3: CoxPH + Encoder+Cox × multi-seed
set -euo pipefail

MYEVAE_ROOT="${MYEVAE_ROOT:-/work/zh/myevae}"
cd "${MYEVAE_ROOT}"

PYTHON="${PYTHON:-python}"
RAW="results/baseline_results_raw.csv"
rm -f "${RAW}"

SEEDS=(42 43 44 45 46)

for SEED in "${SEEDS[@]}"; do
  echo "===== baseline seed=${SEED} ====="
  "${PYTHON}" scripts/run_encoder_cox.py \
    --config configs/baselines.yaml --seed "${SEED}" \
    --output "${RAW}"

  "${PYTHON}" scripts/run_encoder_cox.py \
    --config configs/baselines.yaml --seed "${SEED}" --run-coxph \
    --output "${RAW}"
done

"${PYTHON}" scripts/aggregate_seeds.py \
  --input "${RAW}" \
  --output results/baseline_results_summary.csv

echo "[OK] Baselines → ${RAW}"
