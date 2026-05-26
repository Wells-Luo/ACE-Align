#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

load_defaults() {
  local file="$1"
  local line key value
  while IFS= read -r line || [[ -n "${line}" ]]; do
    [[ -z "${line}" || "${line}" =~ ^[[:space:]]*# ]] && continue
    key="${line%%=*}"
    value="${line#*=}"
    value="${value%\"}"
    value="${value#\"}"
    if [[ -z "${!key:-}" ]]; then
      export "${key}=${value}"
    fi
  done < "${file}"
}

if [[ -f configs/default.env ]]; then
  load_defaults configs/default.env
fi

BASE_MODEL="${BASE_MODEL:?Set BASE_MODEL in configs/default.env or the environment.}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${BASE_MODEL}}"
TRAIN_DATA="${TRAIN_DATA:-data/train/all_countries.jsonl}"
MODEL_NAME="${MODEL_NAME:-causal}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/${MODEL_NAME}}"
LOG_FILE="${LOG_FILE:-logs/train.log}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
GPUS="${GPUS:-0}"

if [[ ! -f "${TRAIN_DATA}" ]]; then
  python scripts/build_all_train.py --output "${TRAIN_DATA}"
fi

CUDA_VISIBLE_DEVICES="${GPUS}" accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --num_machines 1 \
  --mixed_precision no \
  --dynamo_backend no \
  src/train.py \
    --train_data "${TRAIN_DATA}" \
    --base_model "${BASE_MODEL}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    --num_epochs "${NUM_EPOCHS:-2}" \
    --batch_size "${BATCH_SIZE:-8}" \
    --learning_rate "${LEARNING_RATE:-2e-5}" \
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS:-1}" \
    --objective_schedule "${OBJECTIVE_SCHEDULE:-anchor,distribution}" \
    --lora_r "${LORA_R:-16}" \
    --lora_alpha "${LORA_ALPHA:-32}" \
    --lora_dropout "${LORA_DROPOUT:-0.05}" \
    --logging_steps "${LOGGING_STEPS:-20}" \
    --log_file "${LOG_FILE}"
