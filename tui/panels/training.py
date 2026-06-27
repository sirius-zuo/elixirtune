# tui/panels/training.py
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

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._poll_timer = None

    def compose(self) -> ComposeResult:
        yield ConfigForm(_TRAIN_FIELDS, id="train-config-form")
        yield Rule()
        yield Button("▶ Train", id="train-btn", disabled=True, variant="success")
        yield Rule()
        yield LogView(id="train-log")
        yield Label("", id="train-progress")
        yield Sparkline([], id="loss-sparkline", summary_function=min)

    def refresh_content(self) -> None:
        if not self.domain:
            return
        ws = Path("workspaces") / self.domain
        status = infer_status(ws)
        self.query_one("#train-btn", Button).disabled = (
            status_order(status) < status_order(Status.PREPARED)
        )
        self._load_sparkline(ws)

    def _load_sparkline(self, ws: Path) -> None:
        metrics_file = ws / "logs" / "training" / "training_metrics.json"
        if metrics_file.exists():
            try:
                metrics = json.loads(metrics_file.read_text())
                self.query_one(Sparkline).data = metrics.get("train_loss", [])
            except (json.JSONDecodeError, OSError):
                pass

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
        if train_loss:
            self.query_one(Sparkline).data = train_loss
        if iters and train_loss:
            tl = f"{train_loss[-1]:.3f}"
            vl = f"{val_loss[-1]:.3f}" if val_loss else "—"
            self.query_one("#train-progress", Label).update(
                f"Iter {iters[-1]}   train: {tl}   val: {vl}"
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "train-btn" and self.domain:
            event.stop()
            event.button.disabled = True
            ws = Path("workspaces") / self.domain
            generate_runtime_configs(ws)
            self._run_train(self.domain)
            self._poll_timer = self.set_interval(2.0, self._poll_metrics)

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
        self.app._rescan()
