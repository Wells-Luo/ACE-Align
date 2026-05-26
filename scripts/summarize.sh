#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

python src/summarize.py \
  --report_dir "${REPORT_DIR:-results/reports/${MODEL_NAME:-causal}}" \
  --output_dir "${OUTPUT_DIR:-results/csv/${MODEL_NAME:-causal}}" \
  --model_name "${MODEL_NAME:-causal}" \
  --min_samples "${MIN_SAMPLES:-10}"
