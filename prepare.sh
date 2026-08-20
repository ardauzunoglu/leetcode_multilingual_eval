lc-eval prepare \
  --dataset newfacade/LeetCodeDataset \
  --split test \
  --languages python \
  --output runs/python_tasks.jsonl \
  --manifest runs/python_tasks.manifest.json

python -m json.tool runs/python_tasks.manifest.json
