import subprocess
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, TextArea


class NewDomainScreen(ModalScreen):
    DEFAULT_CSS = """
    NewDomainScreen {
        align: center middle;
    }
    NewDomainScreen #dialog {
        width: 60;
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }
    NewDomainScreen #dialog Label {
        height: 1;
        margin-top: 1;
        color: $text-muted;
    }
    NewDomainScreen #dialog Label#dialog-title {
        color: $accent;
        text-style: bold;
        margin-top: 0;
    }
    NewDomainScreen #dialog RadioSet {
        height: auto;
        margin: 1 0;
        border: none;
    }
    NewDomainScreen #dialog TextArea {
        height: 4;
        margin-bottom: 1;
    }
    NewDomainScreen #btn-row {
        height: auto;
        margin-top: 1;
    }
    NewDomainScreen #btn-row Button { width: 1fr; margin-right: 1; }
    NewDomainScreen .hidden { display: none; }
    """

    def __init__(self, root: Path = Path("."), **kwargs) -> None:
        super().__init__(**kwargs)
        self._root = root

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("New Domain", id="dialog-title")
            yield Label("Domain name (lowercase, no spaces):")
            yield Input(id="new-domain-name", placeholder="e.g. my_assistant")
            yield RadioSet(
                RadioButton("Bootstrap from description", id="rb-bootstrap", value=True),
                RadioButton("Import from file", id="rb-import"),
                id="source-radio",
            )
            yield Label("Description:", id="label-desc")
            yield TextArea(id="new-domain-desc")
            yield Label("Seeds file path:", id="label-seeds", classes="hidden")
            yield Input(id="new-domain-seeds-path", placeholder="/path/to/seeds.jsonl",
                        classes="hidden")
            with Horizontal(id="btn-row"):
                yield Button("Create", id="new-domain-create", variant="primary")
                yield Button("Cancel", id="new-domain-cancel")

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        is_import = event.pressed.id == "rb-import"
        self.query_one("#label-desc").set_class(is_import, "hidden")
        self.query_one("#new-domain-desc").set_class(is_import, "hidden")
        self.query_one("#label-seeds").set_class(not is_import, "hidden")
        self.query_one("#new-domain-seeds-path").set_class(not is_import, "hidden")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new-domain-cancel":
            self.dismiss(None)
        elif event.button.id == "new-domain-create":
            name = self.query_one("#new-domain-name", Input).value.strip()
            if not name or " " in name:
                self.app.notify("Enter a valid domain name (no spaces).", severity="warning")
                return
            radio = self.query_one(RadioSet)
            pressed = radio.pressed_button and radio.pressed_button.id
            if pressed == "rb-import":
                seeds = self.query_one("#new-domain-seeds-path", Input).value.strip()
                cmd = ["python3", "cli.py", "init", name, "--seeds", seeds]
            else:
                desc = self.query_one("#new-domain-desc", TextArea).text.strip() or f"{name} domain"
                cmd = ["python3", "cli.py", "init", name, "--desc", desc]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                self.dismiss({"name": name, "success": True})
            else:
                self.dismiss({"name": name, "success": False, "error": result.stderr})
