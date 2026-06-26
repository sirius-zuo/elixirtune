import json
import subprocess
from pathlib import Path

from textual.app import ComposeResult
from textual.widgets import Button, Label, Rule, Sparkline
from textual import work

from tui.app import BasePanel
from tui.domain import infer_status, Status, generate_runtime_configs, status_order
from tui.runner import RunnerOutput, RunnerDone
from tui.widgets.config_form import ConfigField, ConfigForm
from tui.widgets.log_view import LogView

_TRAIN_FIELDS = [
    ConfigField("Base model", "config/model_config.yaml", ["base_model", "path"]),
    ConfigField("LoRA layers", "config/model_config.yaml", ["lora", "num_layers"]),
    ConfigField("Learning rate", "config/training_config.yaml", ["training", "learning_rate"]),
    ConfigField("Iterations", "config/training_config.yaml", ["training", "iters"]),
]


class TrainingPanel(BasePanel):
    DEFAULT_CSS = "TrainingPanel { height: 100%; padding: 1; }"

    def compose(self) -> ComposeResult:
        yield ConfigForm(_TRAIN_FIELDS, id="train-config-form")
        yield Rule()
        yield Button("▶ Train", id="train-btn", disabled=True, variant="success")
        yield Rule()
        yield LogView(id="train-log")
        yield Sparkline([], id="loss-sparkline", summary_function=min)

    def refresh_content(self) -> None:
        if not self.domain:
            return
        ws = Path("workspaces") / self.domain
        status = infer_status(ws)
        self.query_one("#train-btn", Button).disabled = status_order(status) < status_order(Status.PREPARED)
        self._load_sparkline(ws)

    def _load_sparkline(self, ws: Path) -> None:
        metrics_file = ws / "logs" / "training" / "training_metrics.json"
        if metrics_file.exists():
            metrics = json.loads(metrics_file.read_text())
            self.query_one(Sparkline).data = metrics.get("train_loss", [])

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "train-btn" and self.domain:
            event.stop()
            event.button.disabled = True
            ws = Path("workspaces") / self.domain
            generate_runtime_configs(ws)
            self._run_train(self.domain)

    @work(thread=True)
    def _run_train(self, domain: str) -> None:
        ws = Path("workspaces") / domain
        cmd = [
            "python3", "scripts/02_train_model.py",
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
        self.query_one("#train-btn", Button).disabled = False
        if event.exit_code != 0:
            self.query_one(LogView).write_line(
                f"[red]Training failed (exit {event.exit_code})[/red]"
            )
        else:
            self.query_one(LogView).write_line("[green]Training complete.[/green]")
        self.refresh_content()
        self.app._rescan()
