import json
import sys
import types

from lc_eval.generate import generate_with_vllm
from lc_eval.schema import write_jsonl


class FakeCandidate:
    def __init__(self, text):
        self.text = text
        self.finish_reason = "stop"
        self.stop_reason = None
        self.token_ids = [1, 2]


class FakeOutput:
    def __init__(self, count):
        self.outputs = [FakeCandidate(f"```python\ndef solve(): return {i}\n```") for i in range(count)]


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.n = kwargs["n"]


class FakeLLM:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def chat(self, conversations, sampling_params, **kwargs):
        return [FakeOutput(sampling_params.n) for _ in conversations]


def test_resumable_generation(tmp_path, monkeypatch):
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.__version__ = "test"
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    tasks = tmp_path / "tasks.jsonl"
    output = tmp_path / "generations.jsonl"
    config = tmp_path / "config.json"
    write_jsonl(
        tasks,
        [{"task_id": "x", "language": "python", "prompt": "implement solve", "question_id": 1}],
    )
    kwargs = dict(
        tasks_path=tasks,
        output_path=output,
        run_config_path=config,
        model="fake/model",
        languages=["python"],
        n=2,
        request_chunk_size=8,
        max_tokens=100,
        temperature=0.2,
        top_p=0.9,
        top_k=-1,
        min_p=0.0,
        repetition_penalty=1.0,
        seed=13,
        tokenizer=None,
        revision=None,
        tokenizer_revision=None,
        dtype="auto",
        quantization=None,
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        gpu_memory_utilization=0.9,
        max_model_len=None,
        max_num_seqs=None,
        cpu_offload_gb=0.0,
        trust_remote_code=False,
        enforce_eager=False,
        enable_prefix_caching=True,
        chat_template=None,
        resume=True,
    )
    first = generate_with_vllm(**kwargs)
    assert first["generated"] == 2
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["sample_id"] for row in rows] == [0, 1]
    assert rows[0]["code"].startswith("def solve")

    second = generate_with_vllm(**kwargs)
    assert second["generated"] == 0
    assert second["already_complete"] == 1

