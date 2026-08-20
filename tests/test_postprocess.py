import json

import pytest

from lc_eval.postprocess import postprocess_generations
from lc_eval.schema import read_jsonl, write_jsonl


def _generation(**overrides):
    row = {
        "task_id": "task-1",
        "language": "python",
        "sample_id": 0,
        "model": "example/model",
        "raw_text": "```python\ndef solve():\n    return 1\n```",
        "finish_reason": "stop",
    }
    row.update(overrides)
    return row


def _run(tmp_path, rows, processor="auto"):
    source = tmp_path / "raw.jsonl"
    output = tmp_path / "processed.jsonl"
    manifest = tmp_path / "manifest.json"
    write_jsonl(source, rows)
    result = postprocess_generations(
        generations_path=source,
        output_path=output,
        manifest_path=manifest,
        processor=processor,
        overwrite=False,
    )
    return list(read_jsonl(output)), result, json.loads(manifest.read_text())


def test_auto_extracts_gpt_oss_harmony_final(tmp_path):
    rows, result, manifest = _run(
        tmp_path,
        [
            _generation(
                model="openai/gpt-oss-20b",
                raw_text=(
                    "analysisWe should reason first."
                    "assistantfinal```python\ndef solve():\n    return 42\n```"
                ),
            )
        ],
    )

    assert rows[0]["code"] == "def solve():\n    return 42\n"
    assert rows[0]["postprocess"] == {
        "processor": "gpt-oss",
        "status": "ok",
        "method": "harmony_final",
    }
    assert result["status_counts"] == {"ok": 1}
    assert manifest == result


def test_harmony_control_tokens_are_removed(tmp_path):
    rows, _, _ = _run(
        tmp_path,
        [
            _generation(
                model="openai/gpt-oss-20b",
                raw_text=(
                    "<|channel|>analysis<|message|>reasoning"
                    "<|channel|>final<|message|>def solve():\n    return 7\n<|return|>"
                ),
            )
        ],
    )

    assert rows[0]["code"] == "def solve():\n    return 7\n"


def test_missing_gpt_oss_final_is_retained_as_failure(tmp_path):
    rows, result, _ = _run(
        tmp_path,
        [
            _generation(
                model="openai/gpt-oss-20b",
                raw_text="analysisThe output token budget ended here",
                finish_reason="length",
            )
        ],
    )

    assert rows[0]["postprocess"]["status"] == "missing_final"
    assert "postprocess failure" in rows[0]["code"]
    assert result["records"] == 1
    assert result["status_counts"] == {"missing_final": 1}


def test_truncated_harmony_final_is_marked_but_preserved(tmp_path):
    rows, result, _ = _run(
        tmp_path,
        [
            _generation(
                model="openai/gpt-oss-20b",
                raw_text="analysisReasoningassistantfinaldef solve():\n    return",
                finish_reason="length",
            )
        ],
    )

    assert rows[0]["code"] == "def solve():\n    return\n"
    assert rows[0]["postprocess"]["status"] == "truncated_final"
    assert result["status_counts"] == {"truncated_final": 1}


def test_plain_processor_extracts_markdown(tmp_path):
    rows, _, _ = _run(tmp_path, [_generation()], processor="plain")

    assert rows[0]["code"] == "def solve():\n    return 1\n"
    assert rows[0]["postprocess"]["processor"] == "plain"


def test_duplicate_generation_keys_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="duplicate generation key"):
        _run(tmp_path, [_generation(), _generation()])


def test_existing_output_requires_overwrite(tmp_path):
    source = tmp_path / "raw.jsonl"
    output = tmp_path / "processed.jsonl"
    write_jsonl(source, [_generation()])
    output.write_text("existing\n")

    with pytest.raises(FileExistsError, match="--overwrite"):
        postprocess_generations(
            generations_path=source,
            output_path=output,
            manifest_path=tmp_path / "manifest.json",
            processor="auto",
            overwrite=False,
        )
