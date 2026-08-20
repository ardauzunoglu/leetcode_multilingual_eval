#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

LANGUAGE="${LANGUAGE:-python}"
MODEL_NAME="${MODEL_NAME:-gpt-oss-20b}"
REASONING_EFFORT="${REASONING_EFFORT:-high}"
if [[ "${REASONING_EFFORT}" == "mid" ]]; then
  REASONING_EFFORT="medium"
fi

case "${LANGUAGE}" in
  python)
    DEFAULT_IMAGE_PATH="containers/python-3.12-slim.sif"
    ;;
  mojo)
    DEFAULT_IMAGE_PATH="containers/lc-eval-mojo.sif"
    ;;
  cpp | c)
    DEFAULT_IMAGE_PATH="containers/lc-eval-mainstream.sif"
    ;;
  *)
    echo "unsupported language: ${LANGUAGE}" >&2
    exit 2
    ;;
esac

TASKS_PATH="${TASKS_PATH:-runs/${LANGUAGE}_tasks.jsonl}"
RUN_DIR="${RUN_DIR:-runs/${MODEL_NAME}/${LANGUAGE}}"
JUDGE_GENERATIONS_PATH="${JUDGE_GENERATIONS_PATH:-${RUN_DIR}/${LANGUAGE}_generations_${REASONING_EFFORT}_judge_rp1p1.jsonl}"
JUDGMENTS_PATH="${JUDGMENTS_PATH:-${RUN_DIR}/${LANGUAGE}_generations_${REASONING_EFFORT}_judgments_rp1p1.jsonl}"
IMAGE_PATH="${IMAGE_PATH:-${DEFAULT_IMAGE_PATH}}"
WORKERS="${WORKERS:-8}"
COMPILE_TIMEOUT="${COMPILE_TIMEOUT:-30}"
RUN_TIMEOUT="${RUN_TIMEOUT:-5}"
MEMORY_MB="${MEMORY_MB:-4096}"

if [[ ! -f "${IMAGE_PATH}" ]]; then
  echo "container image not found: ${IMAGE_PATH}" >&2
  exit 2
fi

./.venv-py313/bin/lc-eval judge \
  --tasks "${TASKS_PATH}" \
  --generations "${JUDGE_GENERATIONS_PATH}" \
  --output "${JUDGMENTS_PATH}" \
  --backend apptainer \
  --image "${LANGUAGE}=${IMAGE_PATH}" \
  --workers "${WORKERS}" \
  --compile-timeout "${COMPILE_TIMEOUT}" \
  --run-timeout "${RUN_TIMEOUT}" \
  --memory-mb "${MEMORY_MB}"
