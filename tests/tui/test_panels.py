import json
import pytest
from pathlib import Path
from textual.app import App, ComposeResult
from textual.widgets import Button
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
        assert "2" in str(label.content)


async def test_overview_run_button_enabled_with_domain(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    ws.mkdir(parents=True)
    import os; os.chdir(tmp_path)
    async with OverviewApp(ws).run_test() as pilot:
        await pilot.pause()
        from textual.widgets import Button
        btn = pilot.app.query_one("#run-all-btn", Button)
        assert not btn.disabled


from tui.panels.synthetic import SyntheticPanel


class SynthApp(App):
    def __init__(self, ws: Path):
        super().__init__()
        self._ws = ws

    def compose(self) -> ComposeResult:
        yield SyntheticPanel(id="panel")

    def on_mount(self) -> None:
        self.query_one(SyntheticPanel).domain = self._ws.name


async def test_generate_button_disabled_without_seeds(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    ws.mkdir(parents=True)
    import os; os.chdir(tmp_path)
    async with SynthApp(ws).run_test() as pilot:
        await pilot.pause()
        from textual.widgets import Button
        btn = pilot.app.query_one("#gen-btn", Button)
        assert btn.disabled


async def test_generate_button_enabled_with_seeds(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "seeds").mkdir(parents=True)
    (ws / "seeds" / "approved.jsonl").write_text('{"conversation":[]}\n')
    import os; os.chdir(tmp_path)
    async with SynthApp(ws).run_test() as pilot:
        await pilot.pause()
        from textual.widgets import Button
        btn = pilot.app.query_one("#gen-btn", Button)
        assert not btn.disabled


async def test_prepare_button_enabled_with_generated(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "generated").mkdir(parents=True)
    (ws / "generated" / "filtered.jsonl").write_text('{"conversation":[]}\n')
    import os; os.chdir(tmp_path)
    async with SynthApp(ws).run_test() as pilot:
        await pilot.pause()
        from textual.widgets import Button
        btn = pilot.app.query_one("#prepare-btn", Button)
        assert not btn.disabled


from tui.panels.training import TrainingPanel


class TrainApp(App):
    def __init__(self, ws: Path):
        super().__init__()
        self._ws = ws

    def compose(self) -> ComposeResult:
        yield TrainingPanel(id="panel")

    def on_mount(self) -> None:
        self.query_one(TrainingPanel).domain = self._ws.name


async def test_train_button_disabled_without_prepared_data(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    ws.mkdir(parents=True)
    import os; os.chdir(tmp_path)
    async with TrainApp(ws).run_test() as pilot:
        await pilot.pause()
        assert pilot.app.query_one("#train-btn", Button).disabled


async def test_train_button_enabled_with_prepared_data(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "processed").mkdir(parents=True)
    (ws / "processed" / "train.json").write_text("[]")
    import os; os.chdir(tmp_path)
    async with TrainApp(ws).run_test() as pilot:
        await pilot.pause()
        assert not pilot.app.query_one("#train-btn", Button).disabled


async def test_training_panel_shows_sparkline_after_run(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "logs" / "training").mkdir(parents=True)
    import json
    metrics = {"train_loss": [2.0, 1.5, 1.0], "val_loss": [2.1, 1.6, 1.1], "iterations": [100, 200, 300]}
    (ws / "logs" / "training" / "training_metrics.json").write_text(json.dumps(metrics))
    import os; os.chdir(tmp_path)
    async with TrainApp(ws).run_test() as pilot:
        await pilot.pause()
        from textual.widgets import Sparkline
        assert pilot.app.query_one(Sparkline) is not None
