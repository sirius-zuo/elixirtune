import json
import re
import subprocess
from pathlib import Path

import yaml
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Label, Rule, Select
from textual import work

from tui.app import BasePanel
from tui.domain import infer_status, Status, generate_runtime_configs, status_order
from tui.runner import RunnerOutput, RunnerDone, stream_subprocess
from tui.widgets.config_form import ConfigField, ConfigForm
from tui.widgets.log_view import LogView
from tui.widgets.section_rule import SectionRule

_EMBED_TRAIN_FIELDS = [
    ConfigField("Base embedding model", "config/model_config.yaml", ["embedding", "base_model"]),
    ConfigField("LoRA rank", "config/model_config.yaml", ["embedding", "lora", "rank"]),
    ConfigField("Loss type", "config/training_config.yaml", ["embedding", "loss_type"]),
    ConfigField("Learning rate", "config/training_config.yaml", ["embedding", "learning_rate"]),
    ConfigField("Iterations", "config/training_config.yaml", ["embedding", "iters"]),
]

_STEP_RE = re.compile(r"[Ss]tep[:\s]+(\d+).*[Ll]oss[:\s]+([\d.]+)")


class EmbeddingTrainingPanel(BasePanel):
    DEFAULT_CSS = """
    EmbeddingTrainingPanel { height: 100%; padding: 1 1 0 1; }
    EmbeddingTrainingPanel #embed-config-form { height: auto; max-height: 40%; overflow-y: auto; }
    EmbeddingTrainingPanel #embed-summary { height: auto; }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._metrics: dict | None = None

    def compose(self) -> ComposeResult:
        yield ConfigForm(_EMBED_TRAIN_FIELDS, id="embed-config-form")
        yield SectionRule("Training Summary")
        yield Label("", id="embed-summary")
        yield SectionRule("Run")
        with Horizontal(classes="btn-row"):
            yield Button("Import data", id="embed-import-btn", disabled=True, variant="success")
            yield Button("Convert from seeds", id="embed-convert-btn", disabled=True, variant="success")
            yield Button("▶ Train bi-encoder", id="embed-train-btn", disabled=True, variant="success")
            yield Button("▶ Train cross-encoder", id="embed-ce-train-btn", disabled=True, variant="success")
        yield SectionRule("Log")
        yield Label("", id="embed-train-progress")
        yield LogView(id="embed-train-log")
        yield Rule()

    def refresh_content(self) -> None:
        if not self.domain:
            return
        ws = Path("workspaces") / self.domain
        status = infer_status(ws)

        prepared = (ws / "processed" / "embedding_train.json").exists()
        trained = status in (Status.TRAINED, Status.CE_TRAINED)
        has_seeds = (ws / "seeds" / "approved.jsonl").exists()

        self.query_one("#embed-import-btn", Button).disabled = False
        self.query_one("#embed-convert-btn", Button).disabled = not has_seeds
        self.query_one("#embed-train-btn", Button).disabled = not prepared
        self.query_one("#embed-ce-train-btn", Button).disabled = not trained

        self._load_summary(ws)

    def _load_summary(self, ws: Path) -> None:
        label = self.query_one("#embed-summary", Label)
        metrics_file = ws / "logs" / "training" / "training_metrics.json"
        if not metrics_file.exists():
            label.update("")
            return
        try:
            m = json.loads(metrics_file.read_text())
            tl = m.get("train_loss", [])
            its = m.get("iterations", [])
            parts = []
            if its:
                parts.append(f"Iters: {its[-1]}")
            if tl:
                parts.append(f"Train loss: {tl[-1]:.4f}")
            label.update("  ·  ".join(parts) if parts else "")
        except (json.JSONDecodeError, OSError):
            label.update("")

    def watch_domain(self, domain: str | None) -> None:
        self._metrics = None
        super().watch_domain(domain)

    def _capture_metric(self, line: str) -> None:
        if self._metrics is None or not self.domain:
            return
        m = _STEP_RE.search(line)
        if not m:
            return
        self._metrics["iterations"].append(int(m.group(1)))
        self._metrics["train_loss"].append(float(m.group(2)))
        mp = Path("workspaces") / self.domain / "logs" / "training" / "training_metrics.json"
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(self._metrics))
        its, tl = self._metrics["iterations"], self._metrics["train_loss"]
        if its and tl:
            self.query_one("#embed-train-progress", Label).update(
                f"Iter {its[-1]}   loss: {tl[-1]:.3f}"
            )

    def on_config_form_saved(self, _: ConfigForm.Saved) -> None:
        self.app.notify("Config saved.")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if not self.domain:
            return

        if btn_id == "embed-import-btn":
            event.stop()
            self.query_one(LogView).write_line(
                "Run: python cli.py prepare-embedding <domain> --mode import --data-file <path>"
            )

        elif btn_id == "embed-convert-btn":
            event.stop()
            event.button.disabled = True
            self._run_cmd(
                ["python3", "cli.py", "prepare-embedding", self.domain, "--mode", "convert"],
                finish_id="embed-convert-btn",
            )

        elif btn_id == "embed-train-btn":
            event.stop()
            event.button.disabled = True
            ws = Path("workspaces") / self.domain
            generate_runtime_configs(ws)
            self._metrics = {"train_loss": [], "iterations": []}
            self.query_one("#embed-train-progress", Label).update("")
            self._run_train(self.domain, "embedding")

        elif btn_id == "embed-ce-train-btn":
            event.stop()
            event.button.disabled = True
            ws = Path("workspaces") / self.domain
            generate_runtime_configs(ws)
            self._metrics = {"train_loss": [], "iterations": []}
            self.query_one("#embed-train-progress", Label).update("")
            self._run_train(self.domain, "cross-encoder")

    @work(thread=True)
    def _run_cmd(self, cmd: list[str], finish_id: str) -> None:
        for line, code in stream_subprocess(cmd):
            if line is not None:
                self.post_message(RunnerOutput(line))
            else:
                self.post_message(RunnerDone(code, tag=finish_id))

    @work(thread=True)
    def _run_train(self, domain: str, method: str) -> None:
        ws = Path("workspaces") / domain
        if method == "cross-encoder":
            train_data = ws / "processed" / "embedding_train.json"
            val_data = ws / "processed" / "embedding_val.json"
        else:
            train_data = ws / "processed" / "embedding_train.json"
            val_data = ws / "processed" / "embedding_val.json"
        val_args = ["--val-data", str(val_data)] if val_data.exists() else []
        cmd = [
            "python3", "cli.py", "train", domain,
            "--method", method,
            "--model-config", str(ws / "runtime_model_config.yaml"),
            "--training-config", str(ws / "runtime_training_config.yaml"),
            "--train-data", str(train_data),
            *val_args,
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
        self._capture_metric(event.line)

    def on_runner_done(self, event: RunnerDone) -> None:
        if event.tag in ("embed-convert-btn",):
            if event.exit_code != 0:
                self.query_one(LogView).write_line(
                    f"[red]Data prep failed (exit {event.exit_code})[/red]"
                )
            else:
                self.query_one(LogView).write_line("[green]Data prepared.[/green]")
            self.refresh_content()
            self.call_later(self.app._rescan)
            return
        self._metrics = None
        self.query_one("#embed-train-btn", Button).disabled = False
        self.query_one("#embed-ce-train-btn", Button).disabled = False
        if event.exit_code != 0:
            self.query_one(LogView).write_line(
                f"[red]Training failed (exit {event.exit_code})[/red]"
            )
        else:
            self.query_one(LogView).write_line("[green]Training complete.[/green]")
        self.refresh_content()
        self.call_later(self.app._rescan)
