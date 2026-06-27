import json
import subprocess
from pathlib import Path

from textual.app import ComposeResult
from textual.widgets import Button, DataTable, Rule
from textual import work

from tui.app import BasePanel
from tui.domain import infer_status, Status, generate_runtime_configs, status_order
from tui.runner import RunnerOutput, RunnerDone
from tui.widgets.log_view import LogView


class EvaluationPanel(BasePanel):
    DEFAULT_CSS = "EvaluationPanel { height: 100%; padding: 1; }"

    def compose(self) -> ComposeResult:
        yield Button("▶ Evaluate", id="eval-btn", disabled=True, variant="success")
        yield Button("▶ Fuse & Evaluate", id="fuse-eval-btn", disabled=True)
        yield Rule()
        dt = DataTable(id="eval-table")
        dt.add_columns("Model", "BERTScore F1", "Word Overlap")
        yield dt
        yield Rule()
        yield LogView(id="eval-log")

    def refresh_content(self) -> None:
        if not self.domain:
            return
        ws = Path("workspaces") / self.domain
        status = infer_status(ws)
        self.query_one("#eval-btn", Button).disabled = status_order(status) < status_order(Status.TRAINED)
        self.query_one("#fuse-eval-btn", Button).disabled = status_order(status) < status_order(Status.TRAINED)
        self._load_results(ws)

    def _load_results(self, ws: Path) -> None:
        results_dir = ws / "logs" / "evaluation"
        dt = self.query_one(DataTable)
        dt.clear()
        if not results_dir.exists():
            return
        for f in sorted(results_dir.glob("*_evaluation.json")):
            result = json.loads(f.read_text())
            metrics = result.get("metrics", {})
            bs = metrics.get("bertscore") or {}
            bert_f1 = f"{bs.get('f1', {}).get('mean', 0):.4f}" if bs else "—"
            wo = metrics.get("word_overlap", {})
            word_ov = f"{wo.get('mean', 0):.4f}" if wo else "—"
            dt.add_row(result.get("model_name", f.stem), bert_f1, word_ov)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if not self.domain:
            return
        event.stop()
        ws = Path("workspaces") / self.domain
        generate_runtime_configs(ws)
        if event.button.id == "eval-btn":
            event.button.disabled = True
            self._run_eval(self.domain)
        elif event.button.id == "fuse-eval-btn":
            event.button.disabled = True
            self._run_fuse_eval(self.domain)

    @work(thread=True)
    def _run_eval(self, domain: str) -> None:
        ws = Path("workspaces") / domain
        cmd = [
            "python3", "cli.py", "evaluate", domain,
            "--eval-config", str(ws / "runtime_eval_config.yaml"),
            "--adapters-path", str(ws / "adapters"),
            "--test-data", str(ws / "processed" / "test.json"),
        ]
        self._stream(cmd)

    @work(thread=True)
    def _run_fuse_eval(self, domain: str) -> None:
        ws = Path("workspaces") / domain
        cmd = [
            "python3", "cli.py", "fuse", domain,
            "--model-config", str(ws / "runtime_model_config.yaml"),
            "--eval-config", str(ws / "runtime_eval_config.yaml"),
            "--test-data", str(ws / "processed" / "test.json"),
            "--adapters-path", str(ws / "adapters"),
            "--output-path", str(ws / "fused"),
        ]
        self._stream(cmd)

    def _stream(self, cmd: list[str]) -> None:
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
        for btn_id in ("#eval-btn", "#fuse-eval-btn"):
            self.query_one(btn_id, Button).disabled = False
        if event.exit_code != 0:
            self.query_one(LogView).write_line(
                f"[red]Evaluation failed (exit {event.exit_code})[/red]"
            )
        else:
            self.query_one(LogView).write_line("[green]Evaluation complete.[/green]")
        self.refresh_content()
        self.app._rescan()
