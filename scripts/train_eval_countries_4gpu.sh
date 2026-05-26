#!/usr/bin/env bash
set -uo pipefail

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
GPU_LIST="${GPU_LIST:-0 1 2 3}"
RUN_NAME="${RUN_NAME:-causal_country_models}"
MODEL_PREFIX="${MODEL_PREFIX:-causal}"
TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-data/train}"
EVAL_DIR="${EVAL_DIR:-data/eval}"
RESULT_DIR="${RESULT_DIR:-results/predictions/${RUN_NAME}}"
REPORT_DIR="${REPORT_DIR:-results/reports/${RUN_NAME}}"
CSV_DIR="${CSV_DIR:-results/csv/${RUN_NAME}}"
RUN_DIR="${RUN_DIR:-logs/country_4gpu_$(date +%Y%m%d_%H%M%S)}"
COUNTRIES="${COUNTRIES:-australia chile egypt ethiopia germany great_britain india japan mexico new_zealand nigeria philippines russia united_states}"
N_ATTRIBUTES="${N_ATTRIBUTES:-1 2 3 4}"

case "${RUN_DIR}" in
  /*) ;;
  *) RUN_DIR="${ROOT}/${RUN_DIR}" ;;
esac

NUM_EPOCHS="${NUM_EPOCHS:-2}"
OBJECTIVE_SCHEDULE="${OBJECTIVE_SCHEDULE:-anchor,distribution}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
LOGGING_STEPS="${LOGGING_STEPS:-20}"
SEED="${SEED:-42}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"

mkdir -p "${RUN_DIR}" "${RESULT_DIR}" "${REPORT_DIR}" "${CSV_DIR}" outputs
read -r -a gpus <<< "${GPU_LIST}"
read -r -a countries <<< "${COUNTRIES}"
num_workers="${#gpus[@]}"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${RUN_DIR}/run.log"
}

train_one_country() {
  local gpu="$1"
  local worker_idx="$2"
  local country="$3"
  local model_dir="${MODEL_PREFIX}_${country}"
  local output_dir="outputs/${model_dir}"
  local train_data="${TRAIN_DATA_DIR}/${country}.jsonl"
  local train_log="${RUN_DIR}/${model_dir}_train.log"
  local eval_log="${RUN_DIR}/${model_dir}_eval.log"
  local port="$((29600 + worker_idx))"

  if [[ ! -f "${train_data}" ]]; then
    echo "[error] missing training data: ${train_data}"
    return 2
  fi

  echo "[train_start] gpu=${gpu} country=${country} output=${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" accelerate launch \
    --main_process_port "${port}" \
    --num_processes 1 \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    src/train.py \
      --train_data "${train_data}" \
      --base_model "${BASE_MODEL}" \
      --tokenizer_path "${TOKENIZER_PATH}" \
      --output_dir "${output_dir}" \
      --num_epochs "${NUM_EPOCHS}" \
      --batch_size "${BATCH_SIZE}" \
      --learning_rate "${LEARNING_RATE}" \
      --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
      --objective_schedule "${OBJECTIVE_SCHEDULE}" \
      --lora_r "${LORA_R}" \
      --lora_alpha "${LORA_ALPHA}" \
      --lora_dropout "${LORA_DROPOUT}" \
      --logging_steps "${LOGGING_STEPS}" \
      --seed "${SEED}" \
      --log_file "${train_log}"
  local train_rc=$?
  if (( train_rc != 0 )); then
    echo "[train_failed] gpu=${gpu} country=${country} rc=${train_rc}"
    return "${train_rc}"
  fi

  echo "[eval_start] gpu=${gpu} country=${country} adapter=${output_dir}/final"
  CUDA_VISIBLE_DEVICES="${gpu}" python src/infer_eval_vllm.py \
    --base_model "${BASE_MODEL}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --adapter_path "${output_dir}/final" \
    --model_name "${MODEL_PREFIX}" \
    --eval_dir "${EVAL_DIR}" \
    --result_dir "${RESULT_DIR}" \
    --report_dir "${REPORT_DIR}" \
    --countries "${country}" \
    --n_attributes ${N_ATTRIBUTES} \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
    --max_model_len "${VLLM_MAX_MODEL_LEN}" \
    --max_lora_rank "${MAX_LORA_RANK}" \
    --skip_complete > "${eval_log}" 2>&1
  local eval_rc=$?
  if (( eval_rc != 0 )); then
    echo "[eval_failed] gpu=${gpu} country=${country} rc=${eval_rc} log=${eval_log}"
    return "${eval_rc}"
  fi

  echo "[done] gpu=${gpu} country=${country}"
}

worker() {
  local gpu="$1"
  local worker_idx="$2"
  local failed=0
  local idx=0
  local country
  for country in "${countries[@]}"; do
    if (( idx % num_workers == worker_idx )); then
      train_one_country "${gpu}" "${worker_idx}" "${country}" || failed=$((failed + 1))
    fi
    idx=$((idx + 1))
  done
  echo "[worker_done] gpu=${gpu} failed=${failed}"
  return "${failed}"
}

log "run_name=${RUN_NAME}"
log "base_model=${BASE_MODEL}"
log "gpus=${GPU_LIST}"
log "countries=${COUNTRIES}"
log "train: batch=${BATCH_SIZE} epochs=${NUM_EPOCHS} objective=${OBJECTIVE_SCHEDULE}"
log "eval: batch=${EVAL_BATCH_SIZE} n_attributes=${N_ATTRIBUTES}"

pids=()
for i in "${!gpus[@]}"; do
  worker "${gpus[$i]}" "${i}" > "${RUN_DIR}/worker_${i}_gpu${gpus[$i]}.out" 2>&1 &
  pids+=("$!")
  log "started worker=${i} gpu=${gpus[$i]} pid=${pids[-1]}"
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=$((failed + 1))
done
log "failed_workers=${failed}"

python src/summarize.py \
  --report_dir "${REPORT_DIR}" \
  --output_dir "${CSV_DIR}" \
  --model_name "${MODEL_PREFIX}" \
  --countries "${countries[@]}" | tee -a "${RUN_DIR}/run.log"

exit "${failed}"
