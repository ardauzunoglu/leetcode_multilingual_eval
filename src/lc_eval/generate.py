"""Resumable offline batch generation with vLLM."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from .languages import canonical_language, extract_code
from .schema import append_jsonl, read_jsonl


SYSTEM_PROMPT = (
    "You are an expert competitive programmer. Follow the requested function interface exactly "
    "and return only source code."
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_existing(path: Path) -> dict[tuple[str, str], set[int]]:
    existing: dict[tuple[str, str], set[int]] = defaultdict(set)
    if not path.exists():
        return existing
    for row in read_jsonl(path):
        key = (str(row["task_id"]), str(row["language"]))
        sample_id = int(row["sample_id"])
        if sample_id in existing[key]:
            raise ValueError(f"duplicate generation key in {path}: {(*key, sample_id)}")
        existing[key].add(sample_id)
    return existing


def generate_with_vllm(
    *,
    tasks_path: Path,
    output_path: Path,
    run_config_path: Path,
    model: str,
    languages: Sequence[str] | None,
    n: int,
    request_chunk_size: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    repetition_penalty: float,
    seed: int,
    reasoning_effort: str | None,
    tokenizer: str | None,
    revision: str | None,
    tokenizer_revision: str | None,
    dtype: str,
    quantization: str | None,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int | None,
    max_num_seqs: int | None,
    cpu_offload_gb: float,
    trust_remote_code: bool,
    enforce_eager: bool,
    enable_prefix_caching: bool,
    chat_template: Path | None,
    resume: bool,
) -> dict[str, Any]:
    try:
        import vllm
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError("Install the inference extra: pip install -e '.[inference]'") from exc

    selected = {canonical_language(item) for item in languages} if languages else None
    tasks = [
        row for row in read_jsonl(tasks_path)
        if selected is None or str(row["language"]) in selected
    ]
    if not tasks:
        raise ValueError("no prepared tasks match the requested languages")

    if output_path.exists() and not resume:
        output_path.unlink()
    existing = _load_existing(output_path) if resume else defaultdict(set)
    missing_by_count: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        key = (str(task["task_id"]), str(task["language"]))
        missing = n - len(existing[key] & set(range(n)))
        if missing > 0:
            missing_by_count[missing].append(task)

    engine_args: dict[str, Any] = {
        "model": model,
        "tokenizer": tokenizer or model,
        "dtype": dtype,
        "tensor_parallel_size": tensor_parallel_size,
        "pipeline_parallel_size": pipeline_parallel_size,
        "gpu_memory_utilization": gpu_memory_utilization,
        "cpu_offload_gb": cpu_offload_gb,
        "trust_remote_code": trust_remote_code,
        "enforce_eager": enforce_eager,
        "enable_prefix_caching": enable_prefix_caching,
    }
    optional_engine = {
        "revision": revision,
        "tokenizer_revision": tokenizer_revision,
        "quantization": quantization,
        "max_model_len": max_model_len,
        "max_num_seqs": max_num_seqs,
    }
    engine_args.update({key: value for key, value in optional_engine.items() if value is not None})

    config = {
        "created_at_unix": time.time(),
        "tasks_path": str(tasks_path.resolve()),
        "tasks_sha256": file_sha256(tasks_path),
        "output_path": str(output_path.resolve()),
        "model": model,
        "languages": sorted(selected) if selected else sorted({str(row["language"]) for row in tasks}),
        "n": n,
        "request_chunk_size": request_chunk_size,
        "sampling": {
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "min_p": min_p,
            "repetition_penalty": repetition_penalty,
            "seed": seed,
        },
        "reasoning_effort": reasoning_effort,
        "engine": engine_args,
        "chat_template": str(chat_template.resolve()) if chat_template else None,
        "chat_template_sha256": file_sha256(chat_template) if chat_template else None,
        "vllm_version": getattr(vllm, "__version__", "unknown"),
    }
    compatibility_keys = (
        "tasks_sha256",
        "model",
        "languages",
        "n",
        "request_chunk_size",
        "sampling",
        "reasoning_effort",
        "engine",
        "chat_template_sha256",
        "vllm_version",
    )
    if resume and output_path.exists():
        if not run_config_path.exists():
            raise ValueError(
                f"refusing to resume {output_path} without its run config {run_config_path}"
            )
        previous = json.loads(run_config_path.read_text(encoding="utf-8"))
        changed = [key for key in compatibility_keys if previous.get(key) != config.get(key)]
        if changed:
            raise ValueError(
                "refusing to mix incompatible generations; changed settings: "
                + ", ".join(changed)
            )
    run_config_path.parent.mkdir(parents=True, exist_ok=True)
    run_config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if not missing_by_count:
        return {"generated": 0, "already_complete": len(tasks), "config": config}

    llm = LLM(**engine_args)
    template = chat_template.read_text(encoding="utf-8") if chat_template else None
    generated = 0
    for missing_count in sorted(missing_by_count):
        group = missing_by_count[missing_count]
        sampling = SamplingParams(
            n=missing_count,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            seed=seed,
        )
        for start in range(0, len(group), request_chunk_size):
            chunk = group[start:start + request_chunk_size]
            conversations = [
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": str(task["prompt"])},
                ]
                for task in chunk
            ]
            kwargs = {"sampling_params": sampling, "use_tqdm": True}
            if template is not None:
                kwargs["chat_template"] = template
            if reasoning_effort is not None:
                kwargs["chat_template_kwargs"] = {
                    "reasoning_effort": reasoning_effort,
                }
            outputs = llm.chat(conversations, **kwargs)
            for task, output in zip(chunk, outputs):
                key = (str(task["task_id"]), str(task["language"]))
                missing_ids = [index for index in range(n) if index not in existing[key]]
                if len(output.outputs) != len(missing_ids):
                    raise RuntimeError(
                        f"vLLM returned {len(output.outputs)} samples; expected {len(missing_ids)}"
                    )
                for sample_id, candidate in zip(missing_ids, output.outputs):
                    raw_text = candidate.text
                    record = {
                        "task_id": task["task_id"],
                        "question_id": task.get("question_id"),
                        "language": task["language"],
                        "sample_id": sample_id,
                        "model": model,
                        "reasoning_effort": reasoning_effort,
                        "raw_text": raw_text,
                        "code": extract_code(raw_text),
                        "finish_reason": getattr(candidate, "finish_reason", None),
                        "stop_reason": getattr(candidate, "stop_reason", None),
                        "token_count": len(getattr(candidate, "token_ids", []) or []),
                    }
                    append_jsonl(output_path, record)
                    existing[key].add(sample_id)
                    generated += 1

    counts = Counter(language for (_, language), ids in existing.items() for _ in ids)
    return {
        "generated": generated,
        "total_samples_by_language": dict(sorted(counts.items())),
        "config": config,
    }
