#!/usr/bin/env bash
# Batch 1 P0-1: A0 (MyeVAE beta=1.0) and A1 (MO-RiskVAE beta=0.33) × multi-seed
set -euo pipefail

MYEVAE_ROOT="${MYEVAE_ROOT:-/work/zh/myevae}"
cd "${MYEVAE_ROOT}"

SEEDS=(42 43 44 45 46)
INPUTS="${INPUTS_DIR:-inputs}"
PYTHON="${PYTHON:-python}"

export PYTHONPATH="${MYEVAE_ROOT}/src:${PYTHONPATH:-}"

mkdir -p output/main_ablation
mkdir -p results

for SEED in "${SEEDS[@]}"; do
  echo "========== seed=${SEED} =========="

  echo "[A0] MyeVAE beta=1.0 seed=${SEED}"
  "${PYTHON}" -m myevae train os \
    -n "A0_myevae_s${SEED}" \
    -i "${INPUTS}" \
    -o "output/main_ablation/A0_s${SEED}" \
    --beta-start 1.0 \
    --beta-end 1.0 \
    --beta-step 0 \
    --seed "${SEED}"

  echo "[A1] MO-RiskVAE beta=0.33 seed=${SEED}"
  "${PYTHON}" -m myevae train os \
    -n "A1_beta033_s${SEED}" \
    -i "${INPUTS}" \
    -o "output/main_ablation/A1_s${SEED}" \
    --beta-start 0.33 \
    --beta-end 0.33 \
    --beta-step 0 \
    --seed "${SEED}"
done

echo ""
echo "[OK] Training finished. Run evaluation:"
echo "  bash scripts/run_batch_eval_main.sh"
