"""Turn raw model generations into judge-ready source code records."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

from .generate import file_sha256
from .languages import canonical_language, extract_code
from .schema import read_jsonl, write_jsonl


PROCESSORS = ("auto", "plain", "gpt-oss")
HARMONY_FINAL_MARKERS = (
    "<|channel|>final<|message|>",
    "assistantfinal",
)
HARMONY_END_PATTERN = re.compile(r"<\|(?:return|end)\|>")
TRUNCATED_FINISH_REASONS = {"length", "max_tokens"}


def _select_processor(row: Mapping[str, Any], raw_text: str, requested: str) -> str:
    if requested != "auto":
        return requested
    model = str(row.get("model") or "").lower()
    if "gpt-oss" in model or any(marker in raw_text for marker in HARMONY_FINAL_MARKERS):
        return "gpt-oss"
    return "plain"


def _extract_harmony_final(raw_text: str) -> str | None:
    matches = [
        (raw_text.rfind(marker), marker)
        for marker in HARMONY_FINAL_MARKERS
        if marker in raw_text
    ]
    if not matches:
        return None
    position, marker = max(matches, key=lambda item: item[0])
    final = raw_text[position + len(marker):]
    return HARMONY_END_PATTERN.split(final, maxsplit=1)[0]


def _failure_placeholder(language: str, reason: str) -> str:
    message = f"lc-eval postprocess failure: {reason}"
    if language in {"c", "cpp"}:
        return f"/* {message} */\n"
    return f"# {message}\n"


def _process_row(row: Mapping[str, Any], requested_processor: str) -> dict[str, Any]:
    required = ("task_id", "language", "sample_id")
    missing = [key for key in required if key not in row]
    if missing:
        raise ValueError(f"generation is missing required fields: {', '.join(missing)}")

    language = canonical_language(str(row["language"]))
    raw_text = str(row.get("raw_text") or "")
    processor = _select_processor(row, raw_text, requested_processor)
    finish_reason = str(row.get("finish_reason") or "").lower()
    status = "ok"

    if processor == "gpt-oss":
        final = _extract_harmony_final(raw_text)
        if final is None or not final.strip():
            status = "missing_final"
            method = "failure_placeholder"
            code = _failure_placeholder(language, "generation ended before a final answer")
        else:
            if finish_reason in TRUNCATED_FINISH_REASONS:
                status = "truncated_final"
            method = "harmony_final"
            code = extract_code(final)
    else:
        response = raw_text or str(row.get("code") or "")
        if not response.strip():
            status = "empty_response"
            method = "failure_placeholder"
            code = _failure_placeholder(language, "empty generation")
        else:
            if finish_reason in TRUNCATED_FINISH_REASONS:
                status = "truncated_response"
            method = "markdown_or_plain"
            code = extract_code(response)

    result = dict(row)
    result["language"] = language
    result["code"] = code
    result["postprocess"] = {
        "processor": processor,
        "status": status,
        "method": method,
    }
    return result


def postprocess_generations(
    *,
    generations_path: Path,
    output_path: Path,
    manifest_path: Path,
    processor: str,
    overwrite: bool,
) -> dict[str, Any]:
    """Postprocess every generation without dropping failed or truncated samples."""

    if processor not in PROCESSORS:
        raise ValueError(f"unknown processor {processor!r}; choose from {', '.join(PROCESSORS)}")
    if generations_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {output_path}; pass --overwrite to replace it")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, int]] = set()
    statuses: Counter[str] = Counter()
    processors: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    finish_reasons: Counter[str] = Counter()

    for row in read_jsonl(generations_path):
        processed = _process_row(row, processor)
        key = (
            str(processed["task_id"]),
            str(processed["language"]),
            int(processed["sample_id"]),
        )
        if key in seen:
            raise ValueError(f"duplicate generation key: {key}")
        seen.add(key)
        metadata = processed["postprocess"]
        statuses[str(metadata["status"])] += 1
        processors[str(metadata["processor"])] += 1
        methods[str(metadata["method"])] += 1
        finish_reasons[str(processed.get("finish_reason") or "unknown")] += 1
        rows.append(processed)

    write_jsonl(output_path, rows)
    manifest = {
        "generations_path": str(generations_path.resolve()),
        "generations_sha256": file_sha256(generations_path),
        "output_path": str(output_path.resolve()),
        "output_sha256": file_sha256(output_path),
        "requested_processor": processor,
        "records": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "processor_counts": dict(sorted(processors.items())),
        "method_counts": dict(sorted(methods.items())),
        "finish_reason_counts": dict(sorted(finish_reasons.items())),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
