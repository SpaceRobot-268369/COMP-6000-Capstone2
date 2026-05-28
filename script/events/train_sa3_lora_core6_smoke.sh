#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SA3_REPO="${SA3_REPO:-/private/tmp/stable-audio-3}"
PYTHON="${PYTHON:-${REPO_ROOT}/acoustic_ai/.venv-audiogen/bin/python}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/mpl}"

DATA_DIR="${LAYER_C_DATA_DIR:-${REPO_ROOT}/acoustic_ai/data/events/layer_c_sa3_horsfields_bronze_cuckoo_core6_smoke/sa3_lora_core6_data}"
SAVE_DIR="${REPO_ROOT}/model/candidates/burger/layer-c-sa3-horsfields-bronze-cuckoo-core6-smoke/lora_checkpoints"

if [[ ! -f "${SA3_REPO}/scripts/train_lora.py" ]]; then
  echo "Missing Stable Audio 3 repo at ${SA3_REPO}" >&2
  echo "Clone it with: git clone https://github.com/Stability-AI/stable-audio-3.git ${SA3_REPO}" >&2
  exit 1
fi

mkdir -p "${MPLCONFIGDIR}" "${SAVE_DIR}"

env MPLCONFIGDIR="${MPLCONFIGDIR}" "${PYTHON}" "${SA3_REPO}/scripts/train_lora.py" \
  --model small-sfx-base \
  --data_dir "${DATA_DIR}" \
  --save_dir "${SAVE_DIR}" \
  --name layer-c-sa3-bronze-cuckoo-core6-smoke \
  --adapter_type dora-rows \
  --rank 8 \
  --lora_alpha 8 \
  --dropout 0.05 \
  --lr 0.0001 \
  --steps "${SA3_STEPS:-300}" \
  --batch_size 1 \
  --duration 8 \
  --checkpoint_every "${SA3_CHECKPOINT_EVERY:-100}" \
  --demo_every "${SA3_DEMO_EVERY:-999999}" \
  --log_every 10 \
  --num_workers "${SA3_NUM_WORKERS:-0}" \
  --logger none \
  --exclude seconds_total
