from tui.app import BasePanel
from textual.app import ComposeResult
from textual.widgets import Label


class OverviewPanel(BasePanel):
    def compose(self) -> ComposeResult:
        yield Label("Overview", id="overview-placeholder")
