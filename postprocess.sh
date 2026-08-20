#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

./.venv-py313/bin/lc-eval postprocess \
  --generations runs/gpt-oss-20b/python_generations.jsonl \
  --output runs/gpt-oss-20b/python_generations_judge.jsonl \
  --processor auto \
  --overwrite
