import json
import pytest
from pathlib import Path
from textual.app import App, ComposeResult
from tui.app import BasePanel
from tui.panels.overview import OverviewPanel


class OverviewApp(App):
    def __init__(self, ws: Path):
        super().__init__()
        self._ws = ws

    def compose(self) -> ComposeResult:
        yield OverviewPanel(id="panel")

    def on_mount(self) -> None:
        self.query_one(OverviewPanel).domain = self._ws.name


async def test_overview_shows_seed_count(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "seeds").mkdir(parents=True)
    (ws / "seeds" / "approved.jsonl").write_text(
        '{"conversation":[]}\n{"conversation":[]}\n'
    )
    import os; os.chdir(tmp_path)
    async with OverviewApp(ws).run_test() as pilot:
        await pilot.pause()
        from textual.widgets import Label
        label = pilot.app.query_one("#overview-status", Label)
        assert "2" in label._Static__content


async def test_overview_run_button_enabled_with_domain(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    ws.mkdir(parents=True)
    import os; os.chdir(tmp_path)
    async with OverviewApp(ws).run_test() as pilot:
        await pilot.pause()
        from textual.widgets import Button
        btn = pilot.app.query_one("#run-all-btn", Button)
        assert not btn.disabled
