import os

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Checkbox, Input, Label


class HFUploadScreen(ModalScreen):
    DEFAULT_CSS = """
    HFUploadScreen { align: center middle; }
    HFUploadScreen > Vertical {
        width: 62; height: auto;
        background: $surface; border: solid $primary; padding: 1 2;
    }
    HFUploadScreen Label { height: 1; }
    HFUploadScreen Button { margin-top: 1; }
    #hf-error { color: red; height: 1; }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Upload Fused Model to HuggingFace Hub")
            yield Label("Repository name (username/repo-name):")
            yield Input(id="hf-repo-name", placeholder="e.g. username/domain-lora")
            yield Checkbox("Private repository", id="hf-private")
            yield Label("HuggingFace token:")
            yield Input(
                id="hf-token",
                password=True,
                value=os.environ.get("HF_TOKEN", ""),
                placeholder="hf_...",
            )
            yield Label("", id="hf-error")
            yield Button("Upload", id="hf-upload-confirm", variant="primary")
            yield Button("Cancel", id="hf-cancel-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "hf-cancel-btn":
            self.dismiss(None)
            return
        if event.button.id != "hf-upload-confirm":
            return

        repo = self.query_one("#hf-repo-name", Input).value.strip()
        token = self.query_one("#hf-token", Input).value.strip()
        private = self.query_one("#hf-private", Checkbox).value
        error_label = self.query_one("#hf-error", Label)

        if not repo:
            error_label.update("Repository name is required.")
            return
        parts = repo.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            error_label.update("Must be in format: username/repo-name.")
            return
        if not token:
            error_label.update("HuggingFace token is required.")
            return

        self.dismiss({"repo": repo, "private": private, "token": token})
