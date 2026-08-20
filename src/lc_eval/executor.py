"""Resource-limited local or container command execution."""

from __future__ import annotations

import os
import resource
import shutil
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence


@dataclass(frozen=True)
class CommandResult:
    command: tuple[str, ...]
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration_seconds: float


DEFAULT_IMAGES = {
    "python": "python:3.12-slim",
    "c": "gcc:14",
    "cpp": "gcc:14",
}


class Executor:
    def __init__(
        self,
        *,
        backend: str,
        images: Mapping[str, str],
        memory_mb: int,
        cpus: float,
        pids_limit: int,
        output_limit_bytes: int,
        allow_unsafe_local: bool,
    ) -> None:
        self.backend = backend
        self.images = {**DEFAULT_IMAGES, **images}
        self.memory_mb = memory_mb
        self.cpus = cpus
        self.pids_limit = pids_limit
        self.output_limit_bytes = output_limit_bytes
        if backend == "local" and not allow_unsafe_local:
            raise ValueError("local execution is unsafe; pass --allow-unsafe-local explicitly")
        executable = {"docker": "docker", "podman": "podman", "apptainer": "apptainer"}.get(backend)
        if executable and shutil.which(executable) is None:
            raise RuntimeError(f"{executable} is not installed or not on PATH")

    def _image(self, language: str) -> str:
        try:
            return self.images[language]
        except KeyError as exc:
            raise ValueError(
                f"no container image configured for {language}; pass --image {language}=IMAGE"
            ) from exc

    def _outer_command(self, workdir: Path, language: str, command: Sequence[str]) -> list[str]:
        if self.backend == "local":
            return list(command)
        image = self._image(language)
        if self.backend in {"docker", "podman"}:
            cidfile = workdir / ".lc-eval-container-id"
            return [
                self.backend,
                "run",
                "--rm",
                "--pull",
                "never",
                "--cidfile",
                str(cidfile),
                "--network",
                "none",
                "--read-only",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges",
                "--pids-limit",
                str(self.pids_limit),
                "--memory",
                f"{self.memory_mb}m",
                "--cpus",
                str(self.cpus),
                "--user",
                f"{os.getuid()}:{os.getgid()}",
                "--env",
                "HOME=/tmp",
                "--env",
                "TMPDIR=/tmp",
                "--env",
                "LANG=C.UTF-8",
                "--mount",
                f"type=bind,src={workdir.resolve()},dst=/work",
                "--workdir",
                "/work",
                "--tmpfs",
                "/tmp:rw,noexec,nosuid,size=256m",
                image,
                *command,
            ]
        if self.backend == "apptainer":
            return [
                "apptainer",
                "exec",
                "--containall",
                "--no-home",
                "--cleanenv",
                "--no-eval",
                "--env",
                "HOME=/tmp",
                "--env",
                "TMPDIR=/tmp",
                "--env",
                "LANG=C.UTF-8",
                "--writable-tmpfs",
                "--net",
                "--network",
                "none",
                "--bind",
                f"{workdir.resolve()}:/work:rw",
                "--cwd",
                "/work",
                image,
                *command,
            ]
        raise AssertionError(self.backend)

    def _local_preexec(self, timeout: float):
        memory = self.memory_mb * 1024 * 1024
        pids = self.pids_limit

        def apply_limits() -> None:
            os.setsid()
            resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
            cpu_seconds = max(1, int(timeout) + 1)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
            resource.setrlimit(resource.RLIMIT_NPROC, (pids, pids))
            resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024 * 1024, 16 * 1024 * 1024))
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

        return apply_limits

    def run(
        self,
        *,
        workdir: Path,
        language: str,
        command: Sequence[str],
        timeout: float,
    ) -> CommandResult:
        outer = self._outer_command(workdir, language, command)
        started = time.monotonic()
        cidfile = workdir / ".lc-eval-container-id"
        cidfile.unlink(missing_ok=True)
        environment = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "HOME": "/tmp",
            "PYTHONHASHSEED": "0",
        }
        if self.backend == "local" and os.environ.get("LD_LIBRARY_PATH"):
            environment["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]
        process = subprocess.Popen(
            outer,
            cwd=workdir if self.backend == "local" else None,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=environment,
            preexec_fn=self._local_preexec(timeout) if self.backend == "local" else os.setsid,
        )
        buffers = {"stdout": bytearray(), "stderr": bytearray()}

        def drain(name: str, stream) -> None:
            while True:
                try:
                    chunk = stream.read(65536)
                except (OSError, ValueError):
                    return
                if not chunk:
                    return
                remaining = self.output_limit_bytes - len(buffers[name])
                if remaining > 0:
                    buffers[name].extend(chunk[:remaining])

        threads = [
            threading.Thread(target=drain, args=("stdout", process.stdout), daemon=True),
            threading.Thread(target=drain, args=("stderr", process.stderr), daemon=True),
        ]
        for thread in threads:
            thread.start()
        timed_out = False
        try:
            exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = None
            if self.backend in {"docker", "podman"} and cidfile.exists():
                container_id = cidfile.read_text(encoding="utf-8", errors="ignore").strip()
                if container_id:
                    subprocess.run(
                        [self.backend, "kill", container_id],
                        stdin=subprocess.DEVNULL,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=5,
                        env=environment,
                        check=False,
                    )
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        finally:
            for thread in threads:
                thread.join(timeout=1)
            for stream in (process.stdout, process.stderr):
                if stream is not None and process.poll() is not None:
                    stream.close()
            for thread in threads:
                thread.join(timeout=1)
            cidfile.unlink(missing_ok=True)

        stdout = bytes(buffers["stdout"]).decode("utf-8", errors="replace")
        stderr = bytes(buffers["stderr"]).decode("utf-8", errors="replace")
        return CommandResult(
            tuple(outer), exit_code, stdout, stderr, timed_out, time.monotonic() - started
        )
