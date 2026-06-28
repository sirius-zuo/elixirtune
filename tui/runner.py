import re
import subprocess
from collections.abc import Iterator
from textual.message import Message

# Strip ANSI/VT100 escape sequences (colors, cursor moves, erase-line, etc.)
_ANSI_RE = re.compile(r'\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))')


class RunnerOutput(Message):
    def __init__(self, line: str) -> None:
        super().__init__()
        self.line = line


class RunnerDone(Message):
    def __init__(self, exit_code: int, tag: str = "") -> None:
        super().__init__()
        self.exit_code = exit_code
        self.tag = tag


def stream_subprocess(cmd: list[str]) -> Iterator[tuple[str | None, int | None]]:
    """Run cmd and yield (line, None) per output line, then (None, exit_code).

    Reads raw bytes so \r (carriage return) is treated as a terminal would:
    it resets the current line buffer, keeping only the final overwrite.
    ANSI escape sequences are stripped before yielding.
    """
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
        buf = ""
        for chunk in iter(lambda: proc.stdout.read(256), b""):
            for char in chunk.decode("utf-8", errors="replace"):
                if char == "\n":
                    line = _ANSI_RE.sub("", buf).rstrip()
                    if line:
                        yield line, None
                    buf = ""
                elif char == "\r":
                    line = _ANSI_RE.sub("", buf).rstrip()
                    if line:
                        yield "\r" + line, None  # prefix signals: replace previous line
                    buf = ""
                else:
                    buf += char
        # flush any trailing content without a final newline
        if buf.strip():
            line = _ANSI_RE.sub("", buf).rstrip()
            if line:
                yield line, None
    yield None, proc.returncode
