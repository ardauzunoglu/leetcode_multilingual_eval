./.venv-py313/bin/lc-eval judge \
  --tasks runs/python_tasks.jsonl \
  --generations runs/gpt-oss-20b/python_generations_judge.jsonl \
  --output runs/gpt-oss-20b/python_judgments.jsonl \
  --backend apptainer \
  --image python=containers/python-3.12-slim.sif \
  --workers 8 \
  --compile-timeout 30 \
  --run-timeout 5 \
  --memory-mb 4096