import json

from lc_eval.dataset import normalize_tests, parse_function, prepare_dataset
from lc_eval.schema import read_jsonl


def test_parse_function_and_named_literals():
    function = parse_function(
        "class Solution:\n"
        "    def solve(self, nums: List[List[int]], enabled: bool) -> List[int]:\n"
        "        "
    )
    assert function.name == "solve"
    assert [arg.name for arg in function.arguments] == ["nums", "enabled"]
    assert function.arguments[0].type.display() == "list[list[int]]"
    tests = normalize_tests(
        {
            "input_output": [
                {"input": "nums=[[1, 2], [3]], enabled=True", "output": "[1, 2, 3]"}
            ]
        },
        function,
        None,
    )
    assert tests == [{"args": [[[1, 2], [3]], True], "expected": [1, 2, 3]}]


def test_prepare_common_tasks(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps(
            {
                "task_id": "add",
                "problem_description": "Add two integers.",
                "starter_code": "class Solution:\n    def add(self, a: int, b: int) -> int:\n        ",
                "input_output": [{"input": "a=1,b=2", "output": "3"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "tasks.jsonl"
    manifest = tmp_path / "manifest.json"
    result = prepare_dataset(
        input_path=source,
        dataset_name="unused",
        split="test",
        languages=["python", "cpp", "c", "mojo"],
        output_path=output,
        manifest_path=manifest,
        max_tasks=None,
        max_tests=None,
        common_only=True,
    )
    rows = list(read_jsonl(output))
    assert [row["language"] for row in rows] == ["python", "cpp", "c", "mojo"]
    assert result["prepared_tasks_by_language"] == {
        "c": 1,
        "cpp": 1,
        "mojo": 1,
        "python": 1,
    }

