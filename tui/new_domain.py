import subprocess
from pathlib import Path

from textual.app import ComposeResult
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, TextArea


class NewDomainScreen(ModalScreen):
    DEFAULT_CSS = """
    NewDomainScreen { align: center middle; }
    NewDomainScreen > * { width: 60; background: $surface; border: solid $primary; padding: 0 1; }
    Label { height: 1; }
    Input { height: 1; }
    RadioSet { height: 2; }
    Button { height: 1; }
    TextArea { height: 1; }
    """

    def __init__(self, root: Path = Path("."), **kwargs) -> None:
        super().__init__(**kwargs)
        self._root = root

    def compose(self) -> ComposeResult:
        yield Label("New Domain")
        yield Label("Domain name (lowercase, no spaces):")
        yield Input(id="new-domain-name", placeholder="e.g. code_review")
        yield RadioSet(
            RadioButton("Bootstrap from description", id="rb-bootstrap", value=True),
            RadioButton("Import from file", id="rb-import"),
            RadioButton("Download code review dataset", id="rb-code-review"),
            id="source-radio",
        )
        yield Label("Description:")
        yield TextArea(id="new-domain-desc")
        yield Label("Seeds file path:")
        yield Input(id="new-domain-seeds-path", placeholder="/path/to/seeds.jsonl")
        yield Button("Create", id="new-domain-create", variant="primary")
        yield Button("Cancel", id="new-domain-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new-domain-cancel":
            self.dismiss(None)
        elif event.button.id == "new-domain-create":
            name = self.query_one("#new-domain-name", Input).value.strip()
            if not name or " " in name:
                return
            radio = self.query_one(RadioSet)
            pressed = radio.pressed_button and radio.pressed_button.id
            if pressed == "rb-import":
                seeds = self.query_one("#new-domain-seeds-path", Input).value.strip()
                cmd = ["python3", "cli.py", "init", name, "--seeds", seeds]
            elif pressed == "rb-code-review":
                cmd = ["python3", "examples/code_review/setup.py", "--domain", name]
            else:  # bootstrap
                desc = self.query_one("#new-domain-desc", TextArea).text.strip() or f"{name} domain"
                cmd = ["python3", "cli.py", "init", name, "--desc", desc]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                self.dismiss({"name": name, "success": True})
            else:
                self.dismiss({"name": name, "success": False, "error": result.stderr})
