from tui.app import BasePanel
from textual.app import ComposeResult
from textual.widgets import Label


class SyntheticPanel(BasePanel):
    def compose(self) -> ComposeResult:
        yield Label("Synthetic Data", id="synth-placeholder")
