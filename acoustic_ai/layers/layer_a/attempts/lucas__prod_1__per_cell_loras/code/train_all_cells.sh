#!/usr/bin/env bash
# Train one LoRA per (season, diel) cell for lucas__mvp_2__per_cell_loras.
#
# Each cell reuses the proven smoke recipe (r=8, alpha=32, 5 epochs, batch 4,
# lr 1e-5, fp16) — identical to lucas__mvp_1_1__spring_night_replica, just run
# 16 times against 16 single-cell manifests. No shared weights => no gradient
# interference between cells.
#
# Run from acoustic_ai/ inside a tmux session:
#   bash layers/layer_a/attempts/lucas__mvp_2__per_cell_loras/code/train_all_cells.sh
#
# Resumable: skips any cell whose output dir already holds an adapter.

set -euo pipefail

ATTEMPT="lucas__mvp_2__per_cell_loras"
CODE="layers/layer_a/attempts/${ATTEMPT}/code"
DATASET="../resources/site_257_bowra-dry-a/mvp2_per_cell_dataset"
OUT_ROOT="../model/candidates/lucas/mvp_2__per_cell_loras"
LOG_ROOT="${HOME}/lucano/logs/mvp_2_per_cell"
PY="./.venv/bin/python"
ACC="./.venv/bin/accelerate"

EPOCHS=5
BATCH=4
LR=1e-5
LORA_R=8
LORA_ALPHA=32

mkdir -p "${LOG_ROOT}"

# 1. Split the shared manifest into 16 per-cell manifests (idempotent).
echo "=== Splitting manifest into per-cell CSVs ==="
${PY} "${CODE}/split_cell_manifests.py" --manifest "${DATASET}/manifest.csv"

# 2. Train each cell.
CELLS=$(ls "${DATASET}"/cell_*.csv | sed 's#.*/cell_##;s#\.csv##' | sort)
TOTAL=$(echo "${CELLS}" | wc -l | tr -d ' ')
echo ""
echo "=== Training ${TOTAL} per-cell LoRAs (r=${LORA_R}, ${EPOCHS} epochs) ==="

i=0
for cell in ${CELLS}; do
  i=$((i + 1))
  out="${OUT_ROOT}/${cell}"
  log="${LOG_ROOT}/${cell}.log"
  if [ -f "${out}/adapter_model.safetensors" ]; then
    echo "[$i/${TOTAL}] ${cell}: SKIP (adapter exists)"
    continue
  fi
  echo "[$i/${TOTAL}] ${cell}: training -> ${out}  (log: ${log})"
  mkdir -p "${out}"
  ${ACC} launch --mixed_precision fp16 \
    "${CODE}/train_audioldm2.py" \
    --manifest_path "${DATASET}/cell_${cell}.csv" \
    --output_dir "${out}" \
    --batch_size "${BATCH}" --num_epochs "${EPOCHS}" --learning_rate "${LR}" \
    --lora_r "${LORA_R}" --lora_alpha "${LORA_ALPHA}" \
    > "${log}" 2>&1
  # Pull the final-epoch losses into the run summary.
  grep -E "Epoch ${EPOCHS}/${EPOCHS}:" "${log}" | tr '\r' '\n' | grep -oE "Epoch.*" | tail -1 \
    || echo "  (no epoch summary found — check ${log})"
done

echo ""
echo "=== Done. Adapters under ${OUT_ROOT}/<cell>/ ==="
ls -1 "${OUT_ROOT}"
