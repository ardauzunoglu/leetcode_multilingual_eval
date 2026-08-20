"""Functional-correctness summaries and unbiased pass@k."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .schema import read_jsonl


def estimate_pass_at_k(n: int, c: int, k: int) -> float | None:
    if k <= 0:
        raise ValueError("k must be positive")
    if not 0 <= c <= n:
        raise ValueError("correct count must be between zero and n")
    if n < k:
        return None
    if n - c < k:
        return 1.0
    probability_all_wrong = 1.0
    for offset in range(k):
        probability_all_wrong *= (n - c - offset) / (n - offset)
    return 1.0 - probability_all_wrong


def summarize(records: Iterable[Mapping[str, Any]], ks: Sequence[int]) -> dict[str, Any]:
    groups: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    seen: set[tuple[str, str, int]] = set()
    for row in records:
        key = (str(row["task_id"]), str(row["language"]), int(row["sample_id"]))
        if key in seen:
            raise ValueError(f"duplicate judgment key: {key}")
        seen.add(key)
        groups[str(row["language"])][str(row["task_id"])].append(row)
    if not groups:
        raise ValueError("judgment file contains no records")

    by_language: dict[str, Any] = {}
    for language, task_groups in sorted(groups.items()):
        statuses = Counter(
            str(row.get("status") or "unknown")
            for task_rows in task_groups.values()
            for row in task_rows
        )
        metrics: dict[str, Any] = {}
        for k in ks:
            estimates = []
            for task_rows in task_groups.values():
                n = len(task_rows)
                c = sum(bool(row.get("passed")) for row in task_rows)
                estimate = estimate_pass_at_k(n, c, k)
                if estimate is not None:
                    estimates.append(estimate)
            metrics[f"pass@{k}"] = (
                sum(estimates) / len(estimates) if estimates else None
            )
            metrics[f"pass@{k}_eligible_tasks"] = len(estimates)
        samples = sum(statuses.values())
        by_language[language] = {
            "tasks": len(task_groups),
            "samples": samples,
            "passed_samples": statuses.get("passed", 0),
            "compile_rate": (
                (samples - statuses.get("compile_error", 0) - statuses.get("compile_timeout", 0))
                / samples
            ),
            "status_counts": dict(sorted(statuses.items())),
            **metrics,
        }

    task_sets = [set(task_groups) for task_groups in groups.values()]
    common_tasks = set.intersection(*task_sets) if task_sets else set()
    return {
        "languages": sorted(groups),
        "common_task_count": len(common_tasks),
        "by_language": by_language,
    }


def score_file(input_path: Path, output_path: Path, ks: Sequence[int]) -> dict[str, Any]:
    result = summarize(read_jsonl(input_path), ks)
    result["input"] = str(input_path.resolve())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
