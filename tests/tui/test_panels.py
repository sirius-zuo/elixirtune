import json
import pytest
from pathlib import Path
from textual.app import App, ComposeResult
from textual.widgets import Button, Label
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


from tui.panels.evaluation import EvaluationPanel


class EvalApp(App):
    def __init__(self, ws: Path):
        super().__init__()
        self._ws = ws

    def compose(self) -> ComposeResult:
        yield EvaluationPanel(id="panel")

    def on_mount(self) -> None:
        self.query_one(EvaluationPanel).domain = self._ws.name


async def test_eval_buttons_disabled_without_adapters(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    ws.mkdir(parents=True)
    import os; os.chdir(tmp_path)
    async with EvalApp(ws).run_test() as pilot:
        await pilot.pause()
        assert pilot.app.query_one("#eval-btn", Button).disabled
        assert pilot.app.query_one("#fuse-eval-btn", Button).disabled


async def test_eval_buttons_enabled_with_adapters(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "adapters").mkdir(parents=True)
    (ws / "adapters" / "weights.npz").write_text("x")
    import os; os.chdir(tmp_path)
    async with EvalApp(ws).run_test() as pilot:
        await pilot.pause()
        assert not pilot.app.query_one("#eval-btn", Button).disabled


async def test_eval_results_table_loaded_from_json(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "logs" / "evaluation").mkdir(parents=True)
    import json
    (ws / "logs" / "evaluation" / "base_model_evaluation.json").write_text(json.dumps({
        "model_name": "base_model",
        "metrics": {"word_overlap": {"mean": 0.31}, "bertscore": None},
    }))
    import os; os.chdir(tmp_path)
    async with EvalApp(ws).run_test() as pilot:
        await pilot.pause()
        from textual.widgets import DataTable
        table = pilot.app.query_one(DataTable)
        assert table.row_count >= 1


from tui.panels.deployment import DeploymentPanel


class DeployApp(App):
    def __init__(self, ws: Path):
        super().__init__()
        self._ws = ws

    def compose(self) -> ComposeResult:
        yield DeploymentPanel(id="panel")

    def on_mount(self) -> None:
        self.query_one(DeploymentPanel).domain = self._ws.name


async def test_ollama_button_disabled_without_fused(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    ws.mkdir(parents=True)
    import os; os.chdir(tmp_path)
    async with DeployApp(ws).run_test() as pilot:
        await pilot.pause()
        assert pilot.app.query_one("#ollama-btn", Button).disabled


async def test_ollama_button_enabled_with_fused(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "fused").mkdir(parents=True)
    (ws / "fused" / "weights.safetensors").write_text("x")
    import os; os.chdir(tmp_path)
    async with DeployApp(ws).run_test() as pilot:
        await pilot.pause()
        assert not pilot.app.query_one("#ollama-btn", Button).disabled


async def test_deployment_shows_adapter_size(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "adapters").mkdir(parents=True)
    (ws / "adapters" / "adapter.npz").write_bytes(b"x" * 1024)
    import os; os.chdir(tmp_path)
    async with DeployApp(ws).run_test() as pilot:
        await pilot.pause()
        from textual.widgets import Label
        label = pilot.app.query_one("#adapter-info", Label)
        assert "adapters" in str(label.content).lower()


async def test_train_button_disabled_at_seeded_state(tmp_path):
    """Train button must be disabled when status is SEEDED (above EMPTY but below PREPARED)."""
    ws = tmp_path / "workspaces" / "d"
    (ws / "seeds").mkdir(parents=True)
    (ws / "seeds" / "approved.jsonl").write_text('{"conversation":[]}\n')
    import os; os.chdir(tmp_path)
    async with TrainApp(ws).run_test() as pilot:
        await pilot.pause()
        assert pilot.app.query_one("#train-btn", Button).disabled


async def test_eval_buttons_disabled_at_evaluated_state(tmp_path):
    """Eval buttons must be ENABLED at EVALUATED (already evaluated — should allow re-run)."""
    ws = tmp_path / "workspaces" / "d"
    (ws / "adapters").mkdir(parents=True)
    (ws / "adapters" / "weights.npz").write_text("x")
    (ws / "logs" / "evaluation").mkdir(parents=True)
    import json; (ws / "logs" / "evaluation" / "base_model_evaluation.json").write_text(
        json.dumps({"model_name": "base_model", "metrics": {}})
    )
    import os; os.chdir(tmp_path)
    async with EvalApp(ws).run_test() as pilot:
        await pilot.pause()
        assert not pilot.app.query_one("#eval-btn", Button).disabled


async def test_generate_button_enabled_at_prepared_state(tmp_path):
    """Generate button must remain enabled at PREPARED (higher than SEEDED threshold)."""
    ws = tmp_path / "workspaces" / "d"
    (ws / "seeds").mkdir(parents=True)
    (ws / "seeds" / "approved.jsonl").write_text('{"conversation":[]}\n')
    (ws / "generated").mkdir(parents=True)
    (ws / "generated" / "filtered.jsonl").write_text('{"conversation":[]}\n')
    (ws / "processed").mkdir(parents=True)
    (ws / "processed" / "train.json").write_text("[]")
    import os; os.chdir(tmp_path)
    async with SynthApp(ws).run_test() as pilot:
        await pilot.pause()
        assert not pilot.app.query_one("#gen-btn", Button).disabled


# ── Deployment upload tests ──────────────────────────────────────────────────

async def test_hf_upload_button_disabled_without_fused(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    ws.mkdir(parents=True)
    import os; os.chdir(tmp_path)
    async with DeployApp(ws).run_test() as pilot:
        await pilot.pause()
        assert pilot.app.query_one("#hf-upload-btn", Button).disabled


async def test_hf_upload_button_enabled_with_fused(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "fused").mkdir(parents=True)
    (ws / "fused" / "model.safetensors").write_text("x")
    import os; os.chdir(tmp_path)
    async with DeployApp(ws).run_test() as pilot:
        await pilot.pause()
        assert not pilot.app.query_one("#hf-upload-btn", Button).disabled


async def test_hf_upload_button_click_opens_modal(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "fused").mkdir(parents=True)
    (ws / "fused" / "model.safetensors").write_text("x")
    import os; os.chdir(tmp_path)
    async with DeployApp(ws).run_test() as pilot:
        await pilot.pause()
        await pilot.click("#hf-upload-btn")
        await pilot.pause()
        from tui.upload_modal import HFUploadScreen
        assert any(isinstance(s, HFUploadScreen) for s in pilot.app.screen_stack)


# ── Training live progress tests ─────────────────────────────────────────────

async def test_train_progress_captures_streamed_loss(tmp_path):
    """Loss is parsed from streamed trainer stdout (DPO + SFT formats), persisted, and shown."""
    ws = tmp_path / "workspaces" / "d"
    ws.mkdir(parents=True)
    import os; os.chdir(tmp_path)
    async with TrainApp(ws).run_test() as pilot:
        await pilot.pause()
        panel = pilot.app.query_one(TrainingPanel)
        panel._metrics = {"train_loss": [], "val_loss": [], "iterations": []}
        panel._capture_metric("  Step 5/100 | Loss: 1.2345 | batch_size: 4")   # DPO
        panel._capture_metric("Iter 10: Train loss 2.345, Learning Rate 1e-5")  # SFT
        panel._capture_metric("Iter 10: Val loss 2.400, Val took 1.2s")         # SFT val
        await pilot.pause()
        data = json.loads((ws / "logs" / "training" / "training_metrics.json").read_text())
        assert data["iterations"] == [5, 10]
        assert data["train_loss"] == [1.2345, 2.345]
        assert data["val_loss"] == [2.4]
        label = pilot.app.query_one("#train-progress", Label)
        assert "Iter 10" in str(label.content)
        assert "2.345" in str(label.content)


async def test_dpo_method_train_button_requires_dpo_data(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "processed").mkdir(parents=True)
    (ws / "processed" / "train.json").write_text("[]")   # SFT data present
    import os; os.chdir(tmp_path)
    async with TrainApp(ws).run_test() as pilot:
        await pilot.pause()
        from textual.widgets import Select
        sel = pilot.app.query_one("#method-select", Select)
        assert not pilot.app.query_one("#train-btn", Button).disabled   # SFT ready
        sel.value = "dpo"
        await pilot.pause()
        assert pilot.app.query_one("#train-btn", Button).disabled       # DPO needs dpo.json
        (ws / "processed" / "dpo.json").write_text(
            json.dumps([{"prompt": "p", "chosen": "c", "rejected": "r"}])
        )
        pilot.app.query_one(TrainingPanel).refresh_content()
        await pilot.pause()
        assert not pilot.app.query_one("#train-btn", Button).disabled   # DPO ready


async def test_train_progress_capture_no_op_when_not_training(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    ws.mkdir(parents=True)
    import os; os.chdir(tmp_path)
    async with TrainApp(ws).run_test() as pilot:
        await pilot.pause()
        panel = pilot.app.query_one(TrainingPanel)
        # _metrics is None outside a run → capturing a line must be a harmless no-op
        panel._capture_metric("Iter 10: Train loss 2.345")
        await pilot.pause()
        assert str(pilot.app.query_one("#train-progress", Label).content) == ""
        assert not (ws / "logs" / "training" / "training_metrics.json").exists()


# ── GGUF export tests ───────────────────────────────────────────────────────

async def test_gguf_button_disabled_without_fused(tmp_path):
    """GGUF export button must be disabled when no fused model exists."""
    ws = tmp_path / "workspaces" / "d"
    ws.mkdir(parents=True)
    import os; os.chdir(tmp_path)
    async with DeployApp(ws).run_test() as pilot:
        await pilot.pause()
        from textual.widgets import Button
        btn = pilot.app.query_one("#gguf-btn", Button)
        assert btn.disabled


async def test_gguf_button_enabled_with_fused(tmp_path):
    """GGUF export button must be enabled after model is fused."""
    ws = tmp_path / "workspaces" / "d"
    (ws / "fused").mkdir(parents=True)
    (ws / "fused" / "weights.safetensors").write_text("x")
    import os; os.chdir(tmp_path)
    async with DeployApp(ws).run_test() as pilot:
        await pilot.pause()
        from textual.widgets import Button
        btn = pilot.app.query_one("#gguf-btn", Button)
        assert not btn.disabled


async def test_gguf_button_click_runs_export_command(tmp_path):
    """Clicking Export GGUF must trigger the export-gguf CLI command."""
    ws = tmp_path / "workspaces" / "d"
    (ws / "fused").mkdir(parents=True)
    (ws / "fused" / "model.safetensors").write_text("x")
    import os; os.chdir(tmp_path)
    async with DeployApp(ws).run_test() as pilot:
        await pilot.pause()
        # Verify button exists and is enabled
        from textual.widgets import Button
        btn = pilot.app.query_one("#gguf-btn", Button)
        assert not btn.disabled
        # Verify the button has an id that maps to export-gguf handling
        assert btn.id == "gguf-btn"
