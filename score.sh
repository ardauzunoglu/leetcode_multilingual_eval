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
JUDGMENTS_PATH="${JUDGMENTS_PATH:-${RUN_DIR}/${LANGUAGE}_generations_${REASONING_EFFORT}_judgments_rp1p1.jsonl}"
METRICS_PATH="${METRICS_PATH:-${RUN_DIR}/${LANGUAGE}_${REASONING_EFFORT}_metrics.json}"
PASS_K="${PASS_K:-1}"

./.venv-py313/bin/lc-eval score \
  --judgments "${JUDGMENTS_PATH}" \
  --output "${METRICS_PATH}" \
  --k "${PASS_K}"
