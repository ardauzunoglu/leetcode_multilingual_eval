#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

LANGUAGE="${LANGUAGE:-python}"
MODEL_NAME="${MODEL_NAME:-gpt-oss-20b}"
MODEL_ID="${MODEL_ID:-openai/${MODEL_NAME}}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
if [[ "${REASONING_EFFORT}" == "mid" ]]; then
  REASONING_EFFORT="medium"
fi

TASKS_PATH="${TASKS_PATH:-runs/${LANGUAGE}_tasks.jsonl}"
RUN_DIR="${RUN_DIR:-runs/${MODEL_NAME}/${LANGUAGE}}"
OUTPUT_PATH="${OUTPUT_PATH:-${RUN_DIR}/${LANGUAGE}_generations_${REASONING_EFFORT}_rp1p1.jsonl}"
NUM_SAMPLES="${NUM_SAMPLES:-1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
TOP_K="${TOP_K:--1}"
MIN_P="${MIN_P:-0.0}"
if [[ -z "${REPETITION_PENALTY:-}" ]]; then
  if [[ "${REASONING_EFFORT}" == "high" ]]; then
    REPETITION_PENALTY="1.1"
  else
    REPETITION_PENALTY="1.0"
  fi
fi
SEED="${SEED:-13}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
./.venv-py313/bin/lc-eval generate \
  --tasks "${TASKS_PATH}" \
  --languages "${LANGUAGE}" \
  --model "${MODEL_ID}" \
  --output "${OUTPUT_PATH}" \
  --n "${NUM_SAMPLES}" \
  --temperature "${TEMPERATURE}" \
  --top-p "${TOP_P}" \
  --top-k "${TOP_K}" \
  --min-p "${MIN_P}" \
  --repetition-penalty "${REPETITION_PENALTY}" \
  --seed "${SEED}" \
  --reasoning-effort "${REASONING_EFFORT}" \
  --max-tokens "${MAX_TOKENS}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
