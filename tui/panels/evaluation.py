from tui.app import BasePanel
from textual.app import ComposeResult
from textual.widgets import Label


class EvaluationPanel(BasePanel):
    def compose(self) -> ComposeResult:
        yield Label("Evaluation", id="eval-placeholder")
