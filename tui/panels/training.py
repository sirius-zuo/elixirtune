# tui/panels/training.py
import json
import subprocess
from pathlib import Path

import yaml
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Label, Rule
from textual import work

from tui.app import BasePanel
from tui.domain import infer_status, Status, generate_runtime_configs, status_order, resolve_adapters_dir
from tui.runner import RunnerOutput, RunnerDone, stream_subprocess
from tui.widgets.config_form import ConfigField, ConfigForm
from tui.widgets.log_view import LogView
from tui.widgets.section_rule import SectionRule

_TRAIN_FIELDS = [
    ConfigField("Base model", "config/model_config.yaml", ["base_model", "path"]),
    ConfigField("LoRA layers", "config/model_config.yaml", ["lora", "num_layers"]),
    ConfigField("Learning rate", "config/training_config.yaml", ["training", "learning_rate"]),
    ConfigField("Iterations", "config/training_config.yaml", ["training", "iters"]),
]


class TrainingPanel(BasePanel):
    DEFAULT_CSS = """
    TrainingPanel { height: 100%; padding: 1 1 0 1; }
    TrainingPanel #train-config-form { height: auto; max-height: 40%; overflow-y: auto; }
    TrainingPanel #train-summary { height: auto; }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._poll_timer = None

    def compose(self) -> ComposeResult:
        yield ConfigForm(_TRAIN_FIELDS, id="train-config-form")
        yield SectionRule("Training History")
        yield Label("", id="train-summary")
        yield SectionRule("Run Training")
        with Horizontal(classes="btn-row"):
            yield Button("Prepare data", id="prepare-data-btn", disabled=True, variant="success")
            yield Button("▶ Train", id="train-btn", disabled=True, variant="success")
        yield SectionRule("Log")
        yield Label("", id="train-progress")
        yield LogView(id="train-log")
        yield Rule()

    def refresh_content(self) -> None:
        if not self.domain:
            return
        ws = Path("workspaces") / self.domain
        status = infer_status(ws)
        self.query_one("#prepare-data-btn", Button).disabled = (
            status_order(status) < status_order(Status.SEEDED)
        )
        self.query_one("#train-btn", Button).disabled = (
            status_order(status) < status_order(Status.PREPARED)
        )
        self._load_training_summary(ws)

    def _load_training_summary(self, ws: Path) -> None:
        label = self.query_one("#train-summary", Label)
        adapter_dir = resolve_adapters_dir(ws)
        if not adapter_dir.exists():
            label.update("")
            return

        # Completed iterations: infer from highest checkpoint filename
        checkpoints = sorted(adapter_dir.glob("[0-9]*_adapters.safetensors"))
        completed = int(checkpoints[-1].stem.split("_")[0]) if checkpoints else None

        # Loss metrics (may not exist for older runs)
        train_loss_final = val_loss_final = None
        metrics_file = ws / "logs" / "training" / "training_metrics.json"
        if metrics_file.exists():
            try:
                m = json.loads(metrics_file.read_text())
                tl = m.get("train_loss", [])
                vl = m.get("val_loss", [])
                train_loss_final = tl[-1] if tl else None
                val_loss_final = vl[-1] if vl else None
            except (json.JSONDecodeError, OSError):
                pass

        # Training hyperparameters from runtime config
        lr = batch_size = iters_target = None
        t_cfg_file = ws / "runtime_training_config.yaml"
        if t_cfg_file.exists():
            try:
                t = yaml.safe_load(t_cfg_file.read_text()).get("training", {})
                lr = t.get("learning_rate")
                batch_size = t.get("batch_size")
                iters_target = t.get("iters")
            except (yaml.YAMLError, OSError):
                pass

        # LoRA config from adapter_config.json
        lora_rank = lora_layers = None
        a_cfg_file = adapter_dir / "adapter_config.json"
        if a_cfg_file.exists():
            try:
                a = json.loads(a_cfg_file.read_text())
                lora_rank = a.get("lora_parameters", {}).get("rank")
                lora_layers = a.get("num_layers")
            except (json.JSONDecodeError, OSError):
                pass

        # Dataset size
        train_size = None
        stats_file = ws / "processed" / "data_stats.json"
        if stats_file.exists():
            try:
                train_size = json.loads(stats_file.read_text()).get("train_size")
            except (json.JSONDecodeError, OSError):
                pass

        parts = []
        if completed is not None:
            target_str = f"/{iters_target}" if iters_target else ""
            parts.append(f"Iters: {completed}{target_str}")
        if train_loss_final is not None:
            parts.append(f"Train loss: {train_loss_final:.4f}")
        if val_loss_final is not None:
            parts.append(f"Val loss: {val_loss_final:.4f}")
        if lr is not None:
            parts.append(f"LR: {lr}")
        if batch_size is not None:
            parts.append(f"Batch: {batch_size}")
        if lora_rank is not None and lora_layers is not None:
            parts.append(f"LoRA r{lora_rank} × {lora_layers}L")
        if train_size is not None:
            parts.append(f"{train_size:,} samples")

        label.update("  ·  ".join(parts) if parts else "")

    def watch_domain(self, domain: str | None) -> None:
        if self._poll_timer is not None:
            self._poll_timer.stop()
            self._poll_timer = None
        super().watch_domain(domain)

    def _poll_metrics(self) -> None:
        if not self.domain:
            return
        ws = Path("workspaces") / self.domain
        metrics_file = ws / "logs" / "training" / "training_metrics.json"
        if not metrics_file.exists():
            return
        try:
            metrics = json.loads(metrics_file.read_text())
        except (json.JSONDecodeError, OSError):
            return
        train_loss = metrics.get("train_loss", [])
        val_loss = metrics.get("val_loss", [])
        iters = metrics.get("iterations", [])
        if iters and train_loss:
            tl = f"{train_loss[-1]:.3f}"
            vl = f"{val_loss[-1]:.3f}" if val_loss else "—"
            self.query_one("#train-progress", Label).update(
                f"Iter {iters[-1]}   train: {tl}   val: {vl}"
            )

    def on_config_form_saved(self, _: ConfigForm.Saved) -> None:
        self.app.notify("Config saved.")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "prepare-data-btn" and self.domain:
            event.stop()
            event.button.disabled = True
            ws = Path("workspaces") / self.domain
            self._run_cmd(
                ["python3", "cli.py", "prepare", self.domain,
                 "--system-prompt", "You are a helpful assistant.",
                 "--out-dir", str(ws / "processed")],
                finish_id="prepare-data-btn",
            )
        elif event.button.id == "train-btn" and self.domain:
            event.stop()
            event.button.disabled = True
            ws = Path("workspaces") / self.domain
            generate_runtime_configs(ws)
            self._run_train(self.domain)
            self._poll_timer = self.set_interval(2.0, self._poll_metrics)

    @work(thread=True)
    def _run_cmd(self, cmd: list[str], finish_id: str) -> None:
        for line, code in stream_subprocess(cmd):
            if line is not None:
                self.post_message(RunnerOutput(line))
            else:
                self.post_message(RunnerDone(code, tag=finish_id))

    @work(thread=True)
    def _run_train(self, domain: str) -> None:
        ws = Path("workspaces") / domain
        cmd = [
            "python3", "cli.py", "train", domain,
            "--model-config", str(ws / "runtime_model_config.yaml"),
            "--training-config", str(ws / "runtime_training_config.yaml"),
            "--train-data", str(ws / "processed" / "train.json"),
            "--val-data", str(ws / "processed" / "val.json"),
        ]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            self.post_message(RunnerOutput(line.rstrip()))
        proc.wait()
        self.post_message(RunnerDone(proc.returncode))

    def on_runner_output(self, event: RunnerOutput) -> None:
        self.query_one(LogView).write_line(event.line)

    def on_runner_done(self, event: RunnerDone) -> None:
        if event.tag == "prepare-data-btn":
            if event.exit_code != 0:
                self.query_one(LogView).write_line(
                    f"[red]Prepare failed (exit {event.exit_code})[/red]"
                )
            else:
                self.query_one(LogView).write_line("[green]Data prepared.[/green]")
            self.refresh_content()
            self.call_later(self.app._rescan)
            return
        if self._poll_timer is not None:
            self._poll_timer.stop()
            self._poll_timer = None
        self._poll_metrics()
        self.query_one("#train-btn", Button).disabled = False
        if event.exit_code != 0:
            self.query_one(LogView).write_line(
                f"[red]Training failed (exit {event.exit_code})[/red]"
            )
        else:
            self.query_one(LogView).write_line("[green]Training complete.[/green]")
        self.refresh_content()
        self.call_later(self.app._rescan)
