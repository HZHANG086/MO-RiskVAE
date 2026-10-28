#!/usr/bin/env bash
# Evaluate all Batch-1 main ablation runs → results/main_ablation_raw.csv
set -euo pipefail

MYEVAE_ROOT="${MYEVAE_ROOT:-/work/zh/myevae}"
cd "${MYEVAE_ROOT}"

PYTHON="${PYTHON:-python}"
RAW="results/main_ablation_raw.csv"
rm -f "${RAW}"

SEEDS=(42 43 44 45 46)

for SEED in "${SEEDS[@]}"; do
  "${PYTHON}" scripts/eval_cindex.py \
    --model-dir "output/main_ablation/A0_s${SEED}" \
    --split valid --variant A0 --model-name MyeVAE --seed "${SEED}" \
    --output "${RAW}" --append

  "${PYTHON}" scripts/eval_cindex.py \
    --model-dir "output/main_ablation/A1_s${SEED}" \
    --split valid --variant A1 --model-name MO-RiskVAE --seed "${SEED}" \
    --output "${RAW}" --append
done

# A4 (Gumbel) — optional if trained
for SEED in "${SEEDS[@]}"; do
  if [[ -d "output/main_ablation/A4_s${SEED}" ]]; then
    "${PYTHON}" scripts/eval_cindex.py \
      --model-dir "output/main_ablation/A4_s${SEED}" \
      --split valid --variant A4 --model-name Gumbel --seed "${SEED}" \
      --output "${RAW}" --append --K 4 \
      --src-dir "${GUMBEL_ROOT:-/work/zh/gumbel-softmax-vae}/myevae/src"
  fi
done

"${PYTHON}" scripts/aggregate_seeds.py \
  --input "${RAW}" \
  --output results/main_ablation_summary.csv

"${PYTHON}" scripts/paired_test.py \
  --input "${RAW}" --a A1 --b A0

echo "[OK] Summary: results/main_ablation_summary.csv"
