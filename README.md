# Standalone multilingual LeetCode evaluation

This project prepares `newfacade/LeetCodeDataset`, generates code with offline
vLLM, executes hidden tests in isolated language runtimes, and reports pass@k.
It does not depend on the LeetCodeDataset evaluator or MultiPL-E.

The pipeline has five independent commands:

```text
LeetCodeDataset ──prepare──> tasks.jsonl
tasks.jsonl ─────generate──> generations.jsonl
generations.jsonl ─postprocess─> judge_generations.jsonl
tasks + judge_generations ─judge─> judgments.jsonl
judgments.jsonl ─────score──> metrics.json
```

Separating generation and judging lets you generate on GPU nodes and judge on
CPU nodes without keeping vLLM or model weights loaded.

## What is evaluated

The source dataset is Python-native. This evaluator extracts the typed method
signature from `starter_code` and literal cases from `input_output`, then
exposes the same problem as a free function in every requested language. Hidden
tests call that function directly, so models are not required to implement a
JSON or stdin parser.

Built-in adapters currently support:

| Language | Types | Runtime command |
|---|---|---|
| Python | scalars, nested lists, fixed tuples, optional values | Python 3 |
| C++ | scalars and nested vectors | C++20 |
| C | scalars and generated nested-array structs | C17 |
| Mojo | scalars and nested `List` values | Mojo 1.0 |

Tasks involving `TreeNode`, `ListNode`, custom objects, variable-length tuples,
or in-place/void outputs are rejected with a reason in the preparation
manifest. Malformed dataset cases—such as expected outputs containing an
exception or timeout—are also rejected. On the v0.3.1 test file inspected in
August 2026, the common Python/C++/C/Mojo subset contains 173 of 228 tasks. The
pipeline computes this rather than hard-coding the count.

Use `--common-only` (the default) for a fair cross-language comparison. Every
language then receives exactly the same task and test-case set.

## Installation

Create an environment for data preparation and vLLM inference:

```bash
cd /scratch/dkhasha1/auzunog1/leetcode_multilingual_eval
python -m venv .venv
source .venv/bin/activate
pip install -e '.[inference]'
```

Judging and scoring use only the Python standard library. They can run from an
editable install without the inference extra:

```bash
pip install -e .
```

Every option is documented under `lc-eval COMMAND --help`.

## 1. Prepare the benchmark

Load directly from Hugging Face:

```bash
lc-eval prepare \
  --dataset newfacade/LeetCodeDataset \
  --split test \
  --languages python cpp c mojo \
  --output runs/tasks.jsonl \
  --manifest runs/tasks.manifest.json
```

For an offline compute node, download the JSONL once and use:

```bash
lc-eval prepare \
  --input-jsonl /path/to/LeetCodeDataset-test.jsonl \
  --languages python cpp c mojo \
  --output runs/tasks.jsonl
```

`tasks.jsonl` contains both public prompts and hidden cases. `generate` sends
only the `prompt` field to vLLM. Do not use the prepared file as in-context
model input through another mechanism.

Inspect `tasks.manifest.json` before running inference. It records accepted
counts and every dataset- or language-level rejection reason.

## 2. Generate with vLLM

Deterministic pass@1 run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 lc-eval generate \
  --tasks runs/tasks.jsonl \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --output runs/generations.jsonl \
  --n 1 \
  --temperature 0 \
  --max-tokens 4096 \
  --tensor-parallel-size 4
```

Sampling run for pass@1/pass@5:

```bash
lc-eval generate \
  --tasks runs/tasks.jsonl \
  --model Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --output runs/generations.jsonl \
  --n 20 \
  --temperature 0.2 \
  --top-p 0.95 \
  --max-tokens 4096 \
  --tensor-parallel-size 4
```

Generation is append-only and resumable. A task is skipped once sample IDs
`0..n-1` are present. The adjacent config JSON records the model, task-file
hash, vLLM version, engine settings, and decoding settings.

Use `--languages mojo` to generate one language at a time while reusing the
same prepared task file.

## 3. Postprocess model responses

Convert raw responses into judge-ready source code before executing them:

```bash
lc-eval postprocess \
  --generations runs/generations.jsonl \
  --output runs/judge_generations.jsonl \
  --processor auto
```

`auto` handles ordinary Markdown/plain-code responses and detects GPT-OSS
Harmony output from the model ID or response markers. It removes analysis and
keeps only the final channel. A truncated GPT-OSS sample that never produced a
final channel is retained as an explicit failing candidate rather than being
dropped, so pass@k is not inflated. The adjacent manifest reports extraction,
finish-reason, and failure counts. Use `--overwrite` to intentionally replace
an existing postprocessed file.

The output records distinguish `ok`, `missing_final`, `truncated_final`,
`truncated_response`, and `empty_response` statuses. All remain judgeable
records; the status is diagnostic metadata, not a filter.

## 4. Build execution images

Build the provided images once on a Docker/Podman machine:

```bash
docker build \
  -f containers/Dockerfile.mainstream \
  -t lc-eval-mainstream:ubuntu24.04 .

docker build \
  -f containers/Dockerfile.mojo \
  -t lc-eval-mojo:1.0.0b2 .
```

The Mojo image pins the current stable beta. Keep the image tag and compiler
version with published results; Mojo syntax and standard-library behavior are
still evolving. Its installation follows the official `pip install mojo`
method.

On a cluster with Apptainer but no Docker daemon, convert prebuilt OCI images:

```bash
apptainer build lc-eval-mainstream.sif docker://YOUR_REGISTRY/lc-eval-mainstream:ubuntu24.04
apptainer build lc-eval-mojo.sif docker://YOUR_REGISTRY/lc-eval-mojo:1.0.0b2
```

## 5. Judge generated code

Docker example:

```bash
lc-eval judge \
  --tasks runs/tasks.jsonl \
  --generations runs/judge_generations.jsonl \
  --output runs/judgments.jsonl \
  --backend docker \
  --image python=lc-eval-mainstream:ubuntu24.04 \
  --image cpp=lc-eval-mainstream:ubuntu24.04 \
  --image c=lc-eval-mainstream:ubuntu24.04 \
  --image mojo=lc-eval-mojo:1.0.0b2 \
  --workers 8 \
  --compile-timeout 30 \
  --run-timeout 5 \
  --memory-mb 4096
```

Apptainer example:

```bash
lc-eval judge \
  --tasks runs/tasks.jsonl \
  --generations runs/judge_generations.jsonl \
  --output runs/judgments.jsonl \
  --backend apptainer \
  --image python=/path/to/lc-eval-mainstream.sif \
  --image cpp=/path/to/lc-eval-mainstream.sif \
  --image c=/path/to/lc-eval-mainstream.sif \
  --image mojo=/path/to/lc-eval-mojo.sif \
  --workers 8
```

The judge records one of `passed`, `wrong_answer`, `compile_error`,
`compile_timeout`, `runtime_error`, `timeout`, or `missing_pass_marker`, plus
bounded stdout/stderr and compile/run timing. It is append-only and resumable.

### Security boundary

Model-generated code is untrusted. Docker/Podman execution disables networking,
drops capabilities, enables `no-new-privileges`, uses a read-only root, runs as
the invoking non-root UID, and applies memory/CPU/PID/time limits. Do not mount
model directories, credentials, datasets, your home directory, or the Docker
socket into an evaluator container.

Apptainer support is included for HPC convenience, with containment, clean
environment, a separate network namespace, and cgroup limits where the site
allows them. Apptainer is commonly designed for trusted HPC users rather than
as a hostile-code boundary. Confirm your site's configuration before relying
on it. For a stronger boundary, judge on a dedicated Docker/Podman worker,
gVisor, nsjail, or a disposable VM.

`--backend local` exists only for development. It is refused unless
`--allow-unsafe-local` is supplied. Never use it for real model generations.

## 6. Compute metrics

```bash
lc-eval score \
  --judgments runs/judgments.jsonl \
  --output runs/metrics.json \
  --k 1 5
```

Metrics include the unbiased pass@k estimate, eligible task counts, compilation
rate, and status counts for each language. A task contributes to pass@k only
when at least `k` samples were judged.

## Smoke test without a model

The repository contains two synthetic tasks and handwritten correct candidates:

```bash
tmpdir="$(mktemp -d)"

lc-eval prepare \
  --input-jsonl examples/smoke_dataset.jsonl \
  --languages python cpp c mojo \
  --output "${tmpdir}/tasks.jsonl"

lc-eval judge \
  --tasks "${tmpdir}/tasks.jsonl" \
  --generations examples/smoke_generations.jsonl \
  --output "${tmpdir}/judgments.jsonl" \
  --backend local \
  --allow-unsafe-local

lc-eval score \
  --judgments "${tmpdir}/judgments.jsonl" \
  --output "${tmpdir}/metrics.json" \
  --k 1
```

If Mojo is not installed on the host, filter `examples/smoke_generations.jsonl`
to Python/C++/C or run the smoke judge with the Mojo container image.

## Reproducibility checklist

For model comparisons, keep fixed:

- The prepared `tasks.jsonl` hash and common task subset.
- Model and tokenizer revisions.
- Chat template and vLLM version.
- Decoding parameters, seed, and sample count.
- Compiler/runtime image digests.
- Compile/run resource limits.

Passing means passing this dataset's generated hidden cases, not acceptance by
LeetCode's private production test suite.
