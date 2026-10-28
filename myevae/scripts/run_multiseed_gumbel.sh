#!/usr/bin/env bash
# Batch 1 P0-1: A4 Gumbel hybrid — separate repo (do not mix with myevae train paths)
set -euo pipefail

GUMBEL_ROOT="${GUMBEL_ROOT:-/work/zh/gumbel-softmax-vae}"
MYEVAE_ROOT="${MYEVAE_ROOT:-/work/zh/myevae}"

SEEDS=(42 43 44 45 46)
INPUTS="${INPUTS_DIR:-${MYEVAE_ROOT}/inputs}"
PYTHON="${PYTHON:-python}"

cd "${GUMBEL_ROOT}"
export PYTHONPATH="${GUMBEL_ROOT}/myevae/src:${PYTHONPATH:-}"

# Adjust CLI flags to match your gumbel repo's train entrypoint if different
mkdir -p "${MYEVAE_ROOT}/output/main_ablation"

for SEED in "${SEEDS[@]}"; do
  OUT="${MYEVAE_ROOT}/output/main_ablation/A4_s${SEED}"
  echo "[A4] Gumbel hybrid seed=${SEED} -> ${OUT}"
  "${PYTHON}" -m myevae train os \
    -n "A4_gumbel_s${SEED}" \
    -i "${INPUTS}" \
    -o "${OUT}" \
    --beta-start 0.33 \
    --beta-end 0.33 \
    --beta-step 0 \
    --seed "${SEED}"
done

echo ""
echo "[OK] Gumbel training done. Evaluate with:"
echo "  python scripts/eval_cindex.py --model-dir output/main_ablation/A4_s42 \\"
echo "    --variant A4 --model-name Gumbel --seed 42 --K 4 \\"
echo "    --output results/main_ablation_raw.csv --append"
