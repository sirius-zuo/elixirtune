from tui.app import BasePanel
from textual.app import ComposeResult
from textual.widgets import Label


class DeploymentPanel(BasePanel):
    def compose(self) -> ComposeResult:
        yield Label("Deployment", id="deploy-placeholder")
