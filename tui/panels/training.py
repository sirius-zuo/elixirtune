from tui.app import BasePanel
from textual.app import ComposeResult
from textual.widgets import Label


class TrainingPanel(BasePanel):
    def compose(self) -> ComposeResult:
        yield Label("Training", id="training-placeholder")
