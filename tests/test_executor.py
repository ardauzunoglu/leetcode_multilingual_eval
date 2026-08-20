from lc_eval.executor import Executor


def local_executor(output_limit=1024):
    return Executor(
        backend="local",
        images={},
        memory_mb=512,
        cpus=1.0,
        pids_limit=64,
        output_limit_bytes=output_limit,
        allow_unsafe_local=True,
    )


def test_output_is_bounded(tmp_path):
    result = local_executor().run(
        workdir=tmp_path,
        language="python",
        command=("python", "-c", "print('x' * 100000)"),
        timeout=5,
    )
    assert result.exit_code == 0
    assert len(result.stdout.encode()) == 1024


def test_timeout(tmp_path):
    result = local_executor().run(
        workdir=tmp_path,
        language="python",
        command=("python", "-c", "import time; time.sleep(10)"),
        timeout=0.1,
    )
    assert result.timed_out
    assert result.exit_code is None

