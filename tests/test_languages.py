import subprocess

import pytest

from lc_eval.dataset import parse_function
from lc_eval.languages import PASS_PLACEHOLDER, get_adapter


ADD_FUNCTION = parse_function(
    "class Solution:\n    def add(self, a: int, b: int) -> int:\n        "
)
ADD_TESTS = [
    {"args": [2, 3], "expected": 5},
    {"args": [-4, 9], "expected": 5},
]


@pytest.mark.parametrize(
    ("language", "candidate"),
    [
        ("python", "def add(a: int, b: int) -> int:\n    return a + b\n"),
        ("cpp", "long long add(long long a, long long b) { return a + b; }\n"),
        ("c", "long long add(long long a, long long b) { return a + b; }\n"),
    ],
)
def test_generated_harness_compiles_and_passes(tmp_path, language, candidate):
    adapter = get_adapter(language)
    program = adapter.make_program(ADD_FUNCTION, ADD_TESTS, candidate)
    token = "LC_EVAL_PASS_TEST"
    (tmp_path / program.filename).write_text(
        program.source.replace(PASS_PLACEHOLDER, token), encoding="utf-8"
    )
    compiled = subprocess.run(program.compile_command, cwd=tmp_path, capture_output=True, text=True)
    assert compiled.returncode == 0, compiled.stderr
    executed = subprocess.run(program.run_command, cwd=tmp_path, capture_output=True, text=True)
    assert executed.returncode == 0, executed.stderr
    assert token in executed.stdout


def test_wrong_answer_fails(tmp_path):
    adapter = get_adapter("python")
    program = adapter.make_program(
        ADD_FUNCTION,
        ADD_TESTS,
        "def add(a: int, b: int) -> int:\n    return a - b\n",
    )
    (tmp_path / program.filename).write_text(
        program.source.replace(PASS_PLACEHOLDER, "PASS"), encoding="utf-8"
    )
    executed = subprocess.run(program.run_command, cwd=tmp_path, capture_output=True, text=True)
    assert executed.returncode != 0
    assert "test 0 failed" in executed.stderr


def test_mojo_harness_shape():
    adapter = get_adapter("mojo")
    program = adapter.make_program(
        ADD_FUNCTION,
        ADD_TESTS,
        "def add(a: Int, b: Int) -> Int:\n    return a + b\n",
    )
    assert "def main() raises:" in program.source
    assert "mojo" in program.compile_command
    assert PASS_PLACEHOLDER in program.source

