CUDA_VISIBLE_DEVICES=0,1,2,3 lc-eval generate \
  --tasks runs/python_tasks.jsonl \
  --languages python \
  --model openai/gpt-oss-20b \
  --output runs/gpt-oss-20b/python_generations.jsonl \
  --n 1 \
  --temperature 0 \
  --max-tokens 32768 \
  --tensor-parallel-size 4
