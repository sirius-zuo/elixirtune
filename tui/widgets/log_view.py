from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Log


class LogView(Widget):
    DEFAULT_CSS = "LogView { height: 1fr; border: solid $surface; }"

    def compose(self) -> ComposeResult:
        yield Log(id="log-output", auto_scroll=True)

    def write_line(self, text: str) -> None:
        self.query_one(Log).write_line(text)

    def clear(self) -> None:
        self.query_one(Log).clear()
