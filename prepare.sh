lc-eval prepare \
  --dataset newfacade/LeetCodeDataset \
  --split test \
  --languages mojo \
  --output runs/mojo_tasks.jsonl \
  --manifest runs/mojo_tasks.manifest.json

python -m json.tool runs/mojo_tasks.manifest.json
