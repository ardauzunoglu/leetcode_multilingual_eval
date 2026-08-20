import pytest

from lc_eval.score import estimate_pass_at_k, summarize


def test_pass_at_k_estimator():
    assert estimate_pass_at_k(10, 0, 1) == 0.0
    assert estimate_pass_at_k(10, 10, 5) == 1.0
    assert estimate_pass_at_k(10, 2, 1) == pytest.approx(0.2)
    assert estimate_pass_at_k(2, 1, 3) is None


def test_summary():
    records = [
        {"language": "python", "task_id": "a", "sample_id": 0, "status": "passed", "passed": True},
        {"language": "python", "task_id": "a", "sample_id": 1, "status": "wrong_answer", "passed": False},
        {"language": "python", "task_id": "b", "sample_id": 0, "status": "compile_error", "passed": False},
        {"language": "python", "task_id": "b", "sample_id": 1, "status": "wrong_answer", "passed": False},
    ]
    result = summarize(records, [1, 2])
    metrics = result["by_language"]["python"]
    assert metrics["pass@1"] == pytest.approx(0.25)
    assert metrics["pass@2"] == pytest.approx(0.5)
    assert metrics["compile_rate"] == pytest.approx(0.75)
