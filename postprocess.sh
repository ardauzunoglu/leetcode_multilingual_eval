#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

LANGUAGE="${LANGUAGE:-python}"
MODEL_NAME="${MODEL_NAME:-gpt-oss-20b}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
if [[ "${REASONING_EFFORT}" == "mid" ]]; then
  REASONING_EFFORT="medium"
fi

RUN_DIR="${RUN_DIR:-runs/${MODEL_NAME}/${LANGUAGE}}"
GENERATIONS_PATH="${GENERATIONS_PATH:-${RUN_DIR}/${LANGUAGE}_generations_${REASONING_EFFORT}_rp1p1.jsonl}"
JUDGE_GENERATIONS_PATH="${JUDGE_GENERATIONS_PATH:-${RUN_DIR}/${LANGUAGE}_generations_${REASONING_EFFORT}_judge_rp1p1.jsonl}"

./.venv-py313/bin/lc-eval postprocess \
  --generations "${GENERATIONS_PATH}" \
  --output "${JUDGE_GENERATIONS_PATH}" \
  --processor auto \
  --overwrite
