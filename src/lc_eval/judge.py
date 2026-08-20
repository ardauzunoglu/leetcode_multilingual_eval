"""Join prepared tasks with generations and execute hidden harnesses."""

from __future__ import annotations

import json
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

from .executor import Executor
from .generate import file_sha256
from .languages import PASS_PLACEHOLDER, extract_code, get_adapter
from .schema import FunctionSpec, append_jsonl, read_jsonl


def _existing_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    result = set()
    for row in read_jsonl(path):
        key = (str(row["task_id"]), str(row["language"]), int(row["sample_id"]))
        if key in result:
            raise ValueError(f"duplicate judgment key in {path}: {key}")
        result.add(key)
    return result


def _judge_one(
    task: Mapping[str, Any],
    generation: Mapping[str, Any],
    executor: Executor,
    compile_timeout: float,
    run_timeout: float,
) -> dict[str, Any]:
    language = str(task["language"])
    adapter = get_adapter(language)
    function = FunctionSpec.from_json(task["function"])
    candidate = str(generation.get("code") or extract_code(str(generation.get("raw_text") or "")))
    program = adapter.make_program(function, list(task["tests"]), candidate)
    token = f"LC_EVAL_PASS_{uuid.uuid4().hex}"
    source = program.source.replace(PASS_PLACEHOLDER, token)
    started = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="lc-eval-") as temp:
        workdir = Path(temp)
        (workdir / program.filename).write_text(source, encoding="utf-8")
        compile_result = executor.run(
            workdir=workdir,
            language=language,
            command=program.compile_command,
            timeout=compile_timeout,
        )
        base = {
            "task_id": task["task_id"],
            "question_id": task.get("question_id"),
            "language": language,
            "sample_id": int(generation["sample_id"]),
            "model": generation.get("model"),
            "compile_seconds": compile_result.duration_seconds,
            "compile_exit_code": compile_result.exit_code,
            "compile_stdout": compile_result.stdout,
            "compile_stderr": compile_result.stderr,
            "run_seconds": None,
            "run_exit_code": None,
            "run_stdout": "",
            "run_stderr": "",
        }
        if compile_result.timed_out:
            return {**base, "status": "compile_timeout", "passed": False, "total_seconds": time.monotonic() - started}
        if compile_result.exit_code != 0:
            return {**base, "status": "compile_error", "passed": False, "total_seconds": time.monotonic() - started}

        run_result = executor.run(
            workdir=workdir,
            language=language,
            command=program.run_command,
            timeout=run_timeout,
        )
        base.update(
            {
                "run_seconds": run_result.duration_seconds,
                "run_exit_code": run_result.exit_code,
                "run_stdout": run_result.stdout,
                "run_stderr": run_result.stderr,
            }
        )
        if run_result.timed_out:
            status = "timeout"
        elif run_result.exit_code == 0 and token in run_result.stdout:
            status = "passed"
        elif "test " in run_result.stderr and "failed" in run_result.stderr:
            status = "wrong_answer"
        elif run_result.exit_code == 0:
            status = "missing_pass_marker"
        else:
            status = "runtime_error"
        return {
            **base,
            "status": status,
            "passed": status == "passed",
            "total_seconds": time.monotonic() - started,
        }


def judge_generations(
    *,
    tasks_path: Path,
    generations_path: Path,
    output_path: Path,
    config_path: Path,
    backend: str,
    images: Mapping[str, str],
    workers: int,
    compile_timeout: float,
    run_timeout: float,
    memory_mb: int,
    cpus: float,
    pids_limit: int,
    output_limit_bytes: int,
    allow_unsafe_local: bool,
    resume: bool,
) -> dict[str, Any]:
    tasks = {}
    for row in read_jsonl(tasks_path):
        key = (str(row["task_id"]), str(row["language"]))
        if key in tasks:
            raise ValueError(f"duplicate prepared task key: {key}")
        tasks[key] = row
    generations = list(read_jsonl(generations_path))
    if output_path.exists() and not resume:
        output_path.unlink()
    existing = _existing_keys(output_path) if resume else set()
    pending = []
    seen_generations: set[tuple[str, str, int]] = set()
    for generation in generations:
        key = (str(generation["task_id"]), str(generation["language"]))
        result_key = (*key, int(generation["sample_id"]))
        if result_key in seen_generations:
            raise ValueError(f"duplicate generation key: {result_key}")
        seen_generations.add(result_key)
        if result_key in existing:
            continue
        if key not in tasks:
            raise ValueError(f"generation has no matching prepared task: {key}")
        pending.append((tasks[key], generation))

    executor = Executor(
        backend=backend,
        images=images,
        memory_mb=memory_mb,
        cpus=cpus,
        pids_limit=pids_limit,
        output_limit_bytes=output_limit_bytes,
        allow_unsafe_local=allow_unsafe_local,
    )
    config = {
        "created_at_unix": time.time(),
        "tasks_path": str(tasks_path.resolve()),
        "tasks_sha256": file_sha256(tasks_path),
        "generations_path": str(generations_path.resolve()),
        "generations_sha256": file_sha256(generations_path),
        "output_path": str(output_path.resolve()),
        "backend": backend,
        "images": dict(sorted(executor.images.items())),
        "workers": workers,
        "compile_timeout": compile_timeout,
        "run_timeout": run_timeout,
        "memory_mb": memory_mb,
        "cpus": cpus,
        "pids_limit": pids_limit,
        "output_limit_bytes": output_limit_bytes,
    }
    compatibility_keys = (
        "tasks_sha256",
        "backend",
        "images",
        "compile_timeout",
        "run_timeout",
        "memory_mb",
        "cpus",
        "pids_limit",
        "output_limit_bytes",
    )
    if resume and output_path.exists():
        if not config_path.exists():
            raise ValueError(
                f"refusing to resume {output_path} without its run config {config_path}"
            )
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        changed = [key for key in compatibility_keys if previous.get(key) != config.get(key)]
        if changed:
            raise ValueError(
                "refusing to mix incompatible judgments; changed settings: "
                + ", ".join(changed)
            )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                _judge_one, task, generation, executor, compile_timeout, run_timeout
            ): (task["task_id"], generation["sample_id"])
            for task, generation in pending
        }
        for future in as_completed(futures):
            result = future.result()
            append_jsonl(output_path, result)
            completed += 1
    return {"judged": completed, "already_complete": len(existing), "config": config}
