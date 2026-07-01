from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, Select

QUANTIZATIONS = ["Q4_K_M", "Q5_K_M", "Q8_0", "f16"]


class GGUFExportScreen(ModalScreen):
    DEFAULT_CSS = """
    GGUFExportScreen { align: center middle; }
    GGUFExportScreen > Vertical {
        width: 62; height: auto;
        background: $surface; border: solid $primary; padding: 1 2;
    }
    GGUFExportScreen Label { height: 1; }
    GGUFExportScreen Button { margin-top: 1; }
    """

    def __init__(self, domain: str) -> None:
        super().__init__()
        self.domain = domain

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Label("Export GGUF")
            yield Label("Quantization:")
            yield Select(
                [(q, q) for q in QUANTIZATIONS],
                value="Q4_K_M",
                allow_blank=False,
                id="gguf-quantization",
            )
            yield Label("Output path (optional):")
            yield Input(
                id="gguf-output-path",
                placeholder=f"workspaces/{self.domain}/fused/{self.domain}.gguf",
            )
            yield Button("Export", id="gguf-export-confirm", variant="primary")
            yield Button("Cancel", id="gguf-modal-cancel")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        if event.button.id == "gguf-modal-cancel":
            self.dismiss(None)
            return
        if event.button.id != "gguf-export-confirm":
            return

        quantization = self.query_one("#gguf-quantization", Select).value
        output_path = self.query_one("#gguf-output-path", Input).value.strip()

        self.dismiss({
            "quantization": quantization,
            "output_path": output_path or None,
        })
