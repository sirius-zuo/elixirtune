import subprocess
from collections.abc import Iterator
from textual.message import Message


class RunnerOutput(Message):
    def __init__(self, line: str) -> None:
        super().__init__()
        self.line = line


class RunnerDone(Message):
    def __init__(self, exit_code: int) -> None:
        super().__init__()
        self.exit_code = exit_code


def stream_subprocess(cmd: list[str]) -> Iterator[tuple[str | None, int | None]]:
    with subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    ) as proc:
        for line in proc.stdout:
            yield line.rstrip(), None
    yield None, proc.returncode
