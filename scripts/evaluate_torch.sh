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
ADAPTER_PATH="${ADAPTER_PATH:-outputs/causal/final}"
MODEL_NAME="${MODEL_NAME:-causal}"
EVAL_DIR="${EVAL_DIR:-data/eval}"
RESULT_DIR="${RESULT_DIR:-results/predictions}"
REPORT_DIR="${REPORT_DIR:-results/reports/${MODEL_NAME}}"
LOG_DIR="${LOG_DIR:-logs/eval_$(date +%Y%m%d_%H%M%S)}"
GPU_LIST="${GPU_LIST:-0}"
BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
N_ATTRIBUTES="${N_ATTRIBUTES:-1 2 3 4}"

mkdir -p "${LOG_DIR}" "${RESULT_DIR}" "${REPORT_DIR}"
mapfile -t countries < <(find "${EVAL_DIR}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
read -r -a gpus <<< "${GPU_LIST}"
num_workers="${#gpus[@]}"

worker() {
  local gpu="$1"
  local worker_idx="$2"
  local assigned=()
  local idx=0
  local country
  for country in "${countries[@]}"; do
    if (( idx % num_workers == worker_idx )); then
      assigned+=("${country}")
    fi
    idx=$((idx + 1))
  done
  if (( ${#assigned[@]} == 0 )); then
    return 0
  fi
  echo "[gpu ${gpu}] ${assigned[*]}"
  CUDA_VISIBLE_DEVICES="${gpu}" python src/infer_eval.py \
    --base_model "${BASE_MODEL}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --adapter_path "${ADAPTER_PATH}" \
    --model_name "${MODEL_NAME}" \
    --eval_dir "${EVAL_DIR}" \
    --result_dir "${RESULT_DIR}" \
    --report_dir "${REPORT_DIR}" \
    --countries "${assigned[@]}" \
    --n_attributes ${N_ATTRIBUTES} \
    --batch_size "${BATCH_SIZE}" \
    --device cuda:0 \
    --torch_dtype bfloat16 \
    --softmax_method selective \
    --skip_complete
}

echo "model=${MODEL_NAME}" | tee "${LOG_DIR}/run.log"
echo "adapter=${ADAPTER_PATH}" | tee -a "${LOG_DIR}/run.log"
echo "countries=${countries[*]}" | tee -a "${LOG_DIR}/run.log"
echo "gpus=${GPU_LIST}" | tee -a "${LOG_DIR}/run.log"

pids=()
for i in "${!gpus[@]}"; do
  worker "${gpus[$i]}" "${i}" >"${LOG_DIR}/gpu${gpus[$i]}.log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=$((failed + 1))
done
echo "failed_workers=${failed}" | tee -a "${LOG_DIR}/run.log"
exit "${failed}"
