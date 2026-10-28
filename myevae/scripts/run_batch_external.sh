#!/usr/bin/env bash
# External inference for trained checkpoints (after Batch-1 training)
set -euo pipefail

MYEVAE_ROOT="${MYEVAE_ROOT:-/work/zh/myevae}"
cd "${MYEVAE_ROOT}"

PYTHON="${PYTHON:-python}"
RAW="results/external_eval_raw.csv"
CONFIG="configs/external_eval.yaml"
SEED="${SEED:-42}"

rm -f "${RAW}"

run_one() {
  local variant="$1"
  local model_name="$2"
  local dir="$3"
  local extra="${4:-}"
  if [[ ! -d "${dir}" ]]; then
    echo "[SKIP] missing ${dir}"
    return
  fi
  echo "=== ${model_name} (${variant}) ==="
  # shellcheck disable=SC2086
  "${PYTHON}" scripts/run_external_eval.py \
    --model-dir "${dir}" \
    --config "${CONFIG}" \
    --variant "${variant}" \
    --model-name "${model_name}" \
    --seed "${SEED}" \
    --output "${RAW}" \
    --append \
    ${extra}
}

run_one A1 MO-RiskVAE "output/main_ablation/A1_s${SEED}"
run_one A0 MyeVAE "output/main_ablation/A0_s${SEED}"
run_one ENC Encoder+Cox "output/baselines/ENC_s${SEED}"

echo "[OK] External results → ${RAW}"
echo "Dry-run all cohort paths:"
echo "  ${PYTHON} scripts/run_external_eval.py --model-dir output/main_ablation/A1_s${SEED} \\"
echo "    --config ${CONFIG} --variant A1 --model-name MO-RiskVAE --seed ${SEED} --dry-run"
