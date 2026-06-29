import shutil
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Select


class DeleteDomainScreen(ModalScreen):
    DEFAULT_CSS = """
    DeleteDomainScreen { align: center middle; }
    DeleteDomainScreen #dialog {
        width: 64;
        height: auto;
        background: $surface;
        border: solid $error;
        padding: 1 2;
    }
    DeleteDomainScreen Label {
        height: 1;
        color: $text-muted;
    }
    DeleteDomainScreen #dialog-title {
        color: $error;
        text-style: bold;
        margin-bottom: 1;
    }
    DeleteDomainScreen Select { margin-top: 1; margin-bottom: 1; }
    DeleteDomainScreen #confirm-row { display: none; height: auto; margin-top: 1; }
    DeleteDomainScreen #confirm-row.visible { display: block; }
    DeleteDomainScreen #confirm-line1 { color: $error; margin-top: 1; }
    DeleteDomainScreen #confirm-line2 { color: $error; margin-bottom: 1; }
    DeleteDomainScreen #btn-row { height: auto; margin-top: 1; }
    DeleteDomainScreen #btn-row Button { width: 1fr; margin-right: 1; }
    DeleteDomainScreen #confirm-btns { height: auto; margin-top: 1; }
    DeleteDomainScreen #confirm-btns Button { width: 1fr; margin-right: 1; }
    """

    def __init__(self, domains: list[str], root: Path = Path("."), **kwargs) -> None:
        super().__init__(**kwargs)
        self._domains = domains
        self._root = root

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("Delete Domain", id="dialog-title")
            yield Label("Select domain to delete:")
            yield Select(
                [(d, d) for d in self._domains],
                prompt="Choose a domain…",
                allow_blank=True,
                id="domain-select",
            )
            with Vertical(id="confirm-row"):
                yield Label("", id="confirm-line1")
                yield Label("This cannot be undone.", id="confirm-line2")
                with Horizontal(id="confirm-btns"):
                    yield Button("Yes, delete", id="btn-confirm-delete", variant="error")
                    yield Button("Cancel", id="btn-cancel-confirm")
            with Horizontal(id="btn-row"):
                yield Button("Delete", id="btn-delete", variant="error", disabled=True)
                yield Button("Cancel", id="btn-cancel")

    def on_select_changed(self, event: Select.Changed) -> None:
        self.query_one("#btn-delete", Button).disabled = event.value is Select.BLANK
        self.query_one("#confirm-row").remove_class("visible")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id

        if bid in ("btn-cancel", "btn-cancel-confirm"):
            self.dismiss(None)

        elif bid == "btn-delete":
            domain = self.query_one("#domain-select", Select).value
            if domain is Select.BLANK:
                return
            self.query_one("#confirm-line1", Label).update(
                f"Delete '{domain}' and all its data?"
            )
            self.query_one("#confirm-row").add_class("visible")
            self.query_one("#btn-row").display = False

        elif bid == "btn-confirm-delete":
            domain = self.query_one("#domain-select", Select).value
            if domain is Select.BLANK:
                return
            ws = self._root / "workspaces" / str(domain)
            if ws.exists():
                shutil.rmtree(ws)
            self.dismiss({"deleted": str(domain)})
