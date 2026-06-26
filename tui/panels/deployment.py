import subprocess
from pathlib import Path

from textual.app import ComposeResult
from textual.widgets import Button, Label, Rule
from textual import work

from tui.app import BasePanel
from tui.domain import infer_status, Status, generate_runtime_configs
from tui.runner import RunnerOutput, RunnerDone
from tui.widgets.log_view import LogView


def _dir_size_mb(p: Path) -> str:
    if not p.exists():
        return "—"
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return f"{total / 1_048_576:.1f} MB"


class DeploymentPanel(BasePanel):
    DEFAULT_CSS = "DeploymentPanel { height: 100%; padding: 1; }"

    def compose(self) -> ComposeResult:
        yield Label("Adapters: —", id="adapter-info")
        yield Label("Fused model: —", id="fused-info")
        yield Rule()
        yield Button("▶ Fuse & Evaluate", id="fuse-btn", disabled=True, variant="success")
        yield Button("Create Ollama Model", id="ollama-btn", disabled=True)
        yield Rule()
        yield LogView(id="deploy-log")

    def refresh_content(self) -> None:
        if not self.domain:
            return
        ws = Path("workspaces") / self.domain
        status = infer_status(ws)
        adapter_dir = ws / "adapters"
        fused_dir = ws / "fused"
        self.query_one("#adapter-info", Label).update(
            f"workspaces/{self.domain}/adapters   {_dir_size_mb(adapter_dir)}"
        )
        self.query_one("#fused-info", Label).update(
            f"workspaces/{self.domain}/fused       {_dir_size_mb(fused_dir)}"
        )
        status_idx = list(Status).index(status)
        self.query_one("#fuse-btn", Button).disabled = status_idx < list(Status).index(Status.TRAINED)
        self.query_one("#ollama-btn", Button).disabled = status_idx < list(Status).index(Status.DEPLOYED)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if not self.domain:
            return
        event.stop()
        ws = Path("workspaces") / self.domain
        generate_runtime_configs(ws)
        if event.button.id == "fuse-btn":
            event.button.disabled = True
            self._run_fuse(self.domain)
        elif event.button.id == "ollama-btn":
            event.button.disabled = True
            self._run_ollama(self.domain)

    @work(thread=True)
    def _run_fuse(self, domain: str) -> None:
        ws = Path("workspaces") / domain
        cmd = [
            "python3", "scripts/04_fuse_and_evaluate.py",
            "--model-config", str(ws / "runtime_model_config.yaml"),
            "--eval-config", str(ws / "runtime_eval_config.yaml"),
            "--test-data", str(ws / "processed" / "test.json"),
            "--adapters-path", str(ws / "adapters"),
            "--output-path", str(ws / "fused"),
        ]
        self._stream(cmd)

    @work(thread=True)
    def _run_ollama(self, domain: str) -> None:
        ws = Path("workspaces") / domain
        modelfile = ws / "Modelfile"
        modelfile.write_text(f"FROM {ws / 'fused'}\n")
        cmd = ["ollama", "create", f"{domain}-lora", "-f", str(modelfile)]
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
        self.query_one("#fuse-btn", Button).disabled = False
        if event.exit_code != 0:
            self.query_one(LogView).write_line(
                f"[red]Failed (exit {event.exit_code})[/red]"
            )
        else:
            self.query_one(LogView).write_line("[green]Done.[/green]")
        self.refresh_content()
        self.app._rescan()
