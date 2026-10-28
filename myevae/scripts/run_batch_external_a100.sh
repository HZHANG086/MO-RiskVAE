#!/usr/bin/env bash
# External inference on A100 (absolute checkpoint paths)
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/hzhang086/MO-RiskVAE}"
MYEVAE_ROOT="${MYEVAE_ROOT:-${PROJECT_ROOT}/myevae}"
CKPT_ROOT="${CKPT_ROOT:-${PROJECT_ROOT}/output}"
MYEVAE_SRC="${MYEVAE_SRC:-${PROJECT_ROOT}/Jump/gumbel-softmax-vae/myevae/src}"
EXTERNAL_DATA_ROOT="${EXTERNAL_DATA_ROOT:-${MYEVAE_ROOT}/external}"

export MYEVAE_ROOT EXTERNAL_DATA_ROOT
export PYTHONPATH="${MYEVAE_SRC}:${PYTHONPATH:-}"

cd "${MYEVAE_ROOT}"

PYTHON="${PYTHON:-python}"
RAW="${RAW:-results/external_eval_raw.csv}"
CONFIG="configs/external_eval.yaml"
SEED="${SEED:-42}"
APPEND="${APPEND:-0}"

if [[ "${APPEND}" != "1" ]]; then
  rm -f "${RAW}"
fi

run_one() {
  local variant="$1"
  local model_name="$2"
  local dir="$3"
  if [[ ! -d "${dir}" ]]; then
    echo "[SKIP] missing ${dir}"
    return
  fi
  echo "=== ${model_name} (${variant}) seed=${SEED} ==="
  "${PYTHON}" scripts/run_external_eval.py \
    --model-dir "${dir}" \
    --config "${CONFIG}" \
    --variant "${variant}" \
    --model-name "${model_name}" \
    --seed "${SEED}" \
    --src-dir "${MYEVAE_SRC}" \
    --output "${RAW}" \
    --append
}

run_one A1 MO-RiskVAE "${CKPT_ROOT}/main_ablation/A1_s${SEED}"
run_one A0 MyeVAE "${CKPT_ROOT}/main_ablation/A0_s${SEED}"
run_one ENC Encoder+Cox "${CKPT_ROOT}/baselines/ENC_s${SEED}"

echo "[OK] External results → ${MYEVAE_ROOT}/${RAW}"
