import re

from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import TextArea

_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')
_MARKUP_RE = re.compile(r'\[/?[^\]]*\]')


class LogView(Widget):
    DEFAULT_CSS = "LogView { height: 1fr; border: solid #0178D4; }"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._lines: list[str] = []
        self._flushed = 0   # number of lines already in the TextArea
        self._dirty = False

    def compose(self) -> ComposeResult:
        yield TextArea("", id="log-output", read_only=True, soft_wrap=False)

    def write_line(self, text: str) -> None:
        plain = _ANSI_RE.sub("", _MARKUP_RE.sub("", text))
        is_overwrite = plain.startswith("\r")
        plain = plain.lstrip("\r")
        if not plain.strip():
            return
        if is_overwrite and self._lines:
            self._lines[-1] = plain
            self._flushed = 0  # replaced a line already in TextArea — force full reload
        else:
            self._lines.append(plain)
        if not self._dirty:
            self._dirty = True
            self.call_later(self._flush)

    def _flush(self) -> None:
        self._dirty = False
        ta = self.query_one(TextArea)
        new_lines = self._lines[self._flushed:]
        if not new_lines:
            return
        if self._flushed == 0:
            ta.load_text("\n".join(new_lines))
        else:
            ta.insert("\n" + "\n".join(new_lines), location=ta.document.end)
        self._flushed = len(self._lines)
        ta.scroll_end(animate=False)

    def clear(self) -> None:
        self._lines.clear()
        self._flushed = 0
        self._dirty = False
        self.query_one(TextArea).load_text("")
