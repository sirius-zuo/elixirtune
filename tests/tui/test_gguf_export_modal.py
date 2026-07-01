import sys
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Input, Select

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tui.gguf_export_modal import GGUFExportScreen


class GGUFModalApp(App):
    def compose(self) -> ComposeResult:
        return iter([])

    def on_mount(self) -> None:
        self.push_screen(GGUFExportScreen("mydomain"))


async def test_gguf_modal_cancel_dismisses():
    async with GGUFModalApp().run_test() as pilot:
        await pilot.click("#gguf-modal-cancel")
        await pilot.pause()
        assert not any(isinstance(s, GGUFExportScreen) for s in pilot.app.screen_stack)


async def test_gguf_modal_default_confirm_returns_defaults():
    result = None

    class CapturingApp(App):
        def compose(self) -> ComposeResult:
            return iter([])

        def on_mount(self) -> None:
            self.push_screen(GGUFExportScreen("mydomain"), callback=self._capture)

        def _capture(self, value):
            nonlocal result
            result = value

    async with CapturingApp().run_test() as pilot:
        await pilot.click("#gguf-export-confirm")
        await pilot.pause(0.3)
        assert result == {"quantization": "Q4_K_M", "output_path": None}


async def test_gguf_modal_custom_values_returned():
    result = None

    class CapturingApp(App):
        def compose(self) -> ComposeResult:
            return iter([])

        def on_mount(self) -> None:
            self.push_screen(GGUFExportScreen("mydomain"), callback=self._capture)

        def _capture(self, value):
            nonlocal result
            result = value

    async with CapturingApp().run_test() as pilot:
        pilot.app.screen.query_one("#gguf-quantization", Select).value = "Q8_0"
        pilot.app.screen.query_one("#gguf-output-path", Input).value = "custom/out.gguf"
        await pilot.click("#gguf-export-confirm")
        await pilot.pause(0.3)
        assert result == {"quantization": "Q8_0", "output_path": "custom/out.gguf"}


async def test_gguf_modal_placeholder_shows_default_path():
    async with GGUFModalApp().run_test() as pilot:
        placeholder = pilot.app.screen.query_one("#gguf-output-path", Input).placeholder
        assert placeholder == "workspaces/mydomain/fused/mydomain.gguf"
