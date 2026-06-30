# tui/panels/training.py
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
from tui.domain import infer_status, Status, generate_runtime_configs, status_order, resolve_adapters_dir
from tui.runner import RunnerOutput, RunnerDone, stream_subprocess
from tui.widgets.config_form import ConfigField, ConfigForm
from tui.widgets.log_view import LogView
from tui.widgets.section_rule import SectionRule

# mlx_tune trainers stream progress to stdout (no transformers callbacks):
#   DPO: "Step 5/100 | Loss: 1.2345"   SFT (mlx_lm): "Iter 5: Train loss 1.234, ..."
_DPO_STEP_RE = re.compile(r"Step (\d+)/\d+\s*\|\s*Loss:\s*([\d.]+)")
_SFT_TRAIN_RE = re.compile(r"Iter (\d+):\s*Train loss\s*([\d.]+)")
_SFT_VAL_RE = re.compile(r"Iter (\d+):\s*Val loss\s*([\d.]+)")

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
    TrainingPanel #method-select { width: 28; margin-bottom: 1; }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._metrics = None   # accumulates parsed loss while a run streams

    def compose(self) -> ComposeResult:
        yield ConfigForm(_TRAIN_FIELDS, id="train-config-form")
        yield SectionRule("Training History")
        yield Label("", id="train-summary")
        yield SectionRule("Run Training")
        yield Select([("SFT", "sft"), ("DPO", "dpo")], value="sft",
                     allow_blank=False, id="method-select")
        with Horizontal(classes="btn-row"):
            yield Button("Prepare data", id="prepare-data-btn", disabled=True, variant="success")
            yield Button("Prepare DPO data", id="prepare-dpo-btn", disabled=True, variant="success")
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
        method = self._method()
        seeded = status_order(status) >= status_order(Status.SEEDED)
        if method == "dpo":
            # DPO needs preference data (produced by the DPO data pipeline).
            train_disabled = not (ws / "processed" / "dpo.json").exists()
        else:
            train_disabled = status_order(status) < status_order(Status.PREPARED)
        self.query_one("#train-btn", Button).disabled = train_disabled
        # DPO data prep is relevant only for DPO, and needs prompts (seeds/generated).
        self.query_one("#prepare-dpo-btn", Button).disabled = not (method == "dpo" and seeded)
        self._load_training_summary(ws)

    def _method(self) -> str:
        value = self.query_one("#method-select", Select).value
        return value if value is not Select.BLANK else "sft"

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "method-select":
            self.refresh_content()

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
        self._metrics = None
        super().watch_domain(domain)

    def _capture_metric(self, line: str) -> None:
        """Parse a streamed trainer line for loss/step and persist + display it."""
        if self._metrics is None or not self.domain:
            return
        m_train = _DPO_STEP_RE.search(line) or _SFT_TRAIN_RE.search(line)
        m_val = _SFT_VAL_RE.search(line)
        changed = False
        if m_train:
            self._metrics["iterations"].append(int(m_train.group(1)))
            self._metrics["train_loss"].append(float(m_train.group(2)))
            changed = True
        if m_val:
            self._metrics["val_loss"].append(float(m_val.group(2)))
            changed = True
        if not changed:
            return
        mp = Path("workspaces") / self.domain / "logs" / "training" / "training_metrics.json"
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(self._metrics))
        its, tl, vl = self._metrics["iterations"], self._metrics["train_loss"], self._metrics["val_loss"]
        if its and tl:
            vl_s = f"{vl[-1]:.3f}" if vl else "—"
            self.query_one("#train-progress", Label).update(
                f"Iter {its[-1]}   train: {tl[-1]:.3f}   val: {vl_s}"
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
                 "--out-dir", str(ws / "processed")],
                finish_id="prepare-data-btn",
            )
        elif event.button.id == "prepare-dpo-btn" and self.domain:
            event.stop()
            event.button.disabled = True
            ws = Path("workspaces") / self.domain
            generate_runtime_configs(ws)
            self._run_cmd(
                ["python3", "cli.py", "prepare-dpo", self.domain,
                 "--model-config", str(ws / "runtime_model_config.yaml")],
                finish_id="prepare-dpo-btn",
            )
        elif event.button.id == "train-btn" and self.domain:
            event.stop()
            event.button.disabled = True
            ws = Path("workspaces") / self.domain
            generate_runtime_configs(ws)
            self._metrics = {"train_loss": [], "val_loss": [], "iterations": []}
            self.query_one("#train-progress", Label).update("")
            self._run_train(self.domain, self._method())

    @work(thread=True)
    def _run_cmd(self, cmd: list[str], finish_id: str) -> None:
        for line, code in stream_subprocess(cmd):
            if line is not None:
                self.post_message(RunnerOutput(line))
            else:
                self.post_message(RunnerDone(code, tag=finish_id))

    @work(thread=True)
    def _run_train(self, domain: str, method: str = "sft") -> None:
        ws = Path("workspaces") / domain
        if method == "dpo":
            train_data = ws / "processed" / "dpo.json"
            val_args = []   # DPO preference data isn't split into a val set here
        else:
            train_data = ws / "processed" / "train.json"
            val_args = ["--val-data", str(ws / "processed" / "val.json")]
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
        if event.tag in ("prepare-data-btn", "prepare-dpo-btn"):
            what = "DPO data" if event.tag == "prepare-dpo-btn" else "Data"
            if event.exit_code != 0:
                self.query_one(LogView).write_line(
                    f"[red]{what} prep failed (exit {event.exit_code})[/red]"
                )
            else:
                self.query_one(LogView).write_line(f"[green]{what} prepared.[/green]")
            self.refresh_content()
            self.call_later(self.app._rescan)
            return
        self._metrics = None
        self.query_one("#train-btn", Button).disabled = False
        if event.exit_code != 0:
            self.query_one(LogView).write_line(
                f"[red]Training failed (exit {event.exit_code})[/red]"
            )
        else:
            self.query_one(LogView).write_line("[green]Training complete.[/green]")
        self.refresh_content()
        self.call_later(self.app._rescan)
