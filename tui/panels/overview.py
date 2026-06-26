import subprocess
from pathlib import Path

from textual import work
from textual.app import ComposeResult
from textual.widgets import Button, Label

from tui.app import BasePanel
from tui.domain import generate_runtime_configs
from tui.runner import RunnerOutput, RunnerDone
from tui.widgets.log_view import LogView


class OverviewPanel(BasePanel):
    DEFAULT_CSS = "OverviewPanel { height: 100%; padding: 1; }"

    def compose(self) -> ComposeResult:
        yield Label("Select a domain.", id="overview-status")
        yield Button("▶ Run Full Pipeline", id="run-all-btn", disabled=True)
        yield LogView(id="overview-log")

    def refresh_content(self) -> None:
        if not self.domain:
            return
        ws = Path("workspaces") / self.domain
        self.query_one("#overview-status", Label).update(self._summary(ws))
        self.query_one("#run-all-btn", Button).disabled = False

    def _summary(self, ws: Path) -> str:
        lines = []
        approved = ws / "seeds" / "approved.jsonl"
        if approved.exists():
            count = sum(1 for _ in approved.open())
            lines.append(f"Seeds: {count} approved")
        filtered = ws / "generated" / "filtered.jsonl"
        if filtered.exists():
            count = sum(1 for _ in filtered.open())
            lines.append(f"Generated: {count} records")
        for split in ("train", "val", "test"):
            p = ws / "processed" / f"{split}.json"
            if p.exists():
                import json
                lines.append(f"  {split}: {len(json.loads(p.read_text()))}")
        adapter = ws / "adapters"
        if adapter.exists() and any(adapter.iterdir()):
            lines.append("Adapter: ready")
        return "\n".join(lines) if lines else f"Workspace: {self.domain}"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "run-all-btn" and self.domain:
            event.stop()
            self.query_one("#run-all-btn", Button).disabled = True
            self._run_pipeline(self.domain)

    @work(thread=True)
    def _run_pipeline(self, domain: str) -> None:
        ws = Path("workspaces") / domain
        generate_runtime_configs(ws)
        steps = [
            ["python3", "cli.py", "generate", domain],
            ["python3", "cli.py", "prepare", domain,
             "--system-prompt", "You are a helpful assistant.",
             "--out-dir", str(ws / "processed")],
            ["python3", "scripts/02_train_model.py",
             "--model-config", str(ws / "runtime_model_config.yaml"),
             "--training-config", str(ws / "runtime_training_config.yaml"),
             "--train-data", str(ws / "processed" / "train.json"),
             "--val-data", str(ws / "processed" / "val.json")],
            ["python3", "scripts/03_evaluate_model.py",
             "--config", str(ws / "runtime_eval_config.yaml"),
             "--adapters-path", str(ws / "adapters"),
             "--test-data", str(ws / "processed" / "test.json")],
        ]
        for cmd in steps:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
            )
            for line in proc.stdout:
                self.post_message(RunnerOutput(line.rstrip()))
            proc.wait()
            if proc.returncode != 0:
                self.post_message(RunnerDone(proc.returncode))
                return
        self.post_message(RunnerDone(0))

    def on_runner_output(self, event: RunnerOutput) -> None:
        self.query_one(LogView).write_line(event.line)

    def on_runner_done(self, event: RunnerDone) -> None:
        self.query_one("#run-all-btn", Button).disabled = False
        if event.exit_code != 0:
            self.query_one(LogView).write_line(
                f"[red]Pipeline failed (exit {event.exit_code})[/red]"
            )
        else:
            self.query_one(LogView).write_line("[green]Pipeline complete.[/green]")
            self.app._rescan()
