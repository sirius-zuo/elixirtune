import subprocess
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Label
from textual import work

from tui.app import BasePanel
from tui.domain import infer_status, Status, status_order
from tui.runner import RunnerOutput, RunnerDone
from tui.widgets.config_form import ConfigField, ConfigForm
from tui.widgets.log_view import LogView
from tui.widgets.section_rule import SectionRule

_SYNTH_FIELDS = [
    ConfigField("Teacher URL", "workspaces/{domain}/config.yaml", ["teacher", "base_url"]),
    ConfigField("Model", "workspaces/{domain}/config.yaml", ["teacher", "model"]),
    ConfigField("API Key", "workspaces/{domain}/config.yaml", ["teacher", "api_key"], password=True),
    ConfigField("Target size", "workspaces/{domain}/config.yaml", ["generate", "target_size"]),
    ConfigField("Judge cutoff", "workspaces/{domain}/config.yaml", ["filter", "judge", "score_cutoff"]),
]


class SyntheticPanel(BasePanel):
    DEFAULT_CSS = "SyntheticPanel { height: 100%; padding: 1; }"

    def compose(self) -> ComposeResult:
        yield ConfigForm(_SYNTH_FIELDS, id="synth-config-form")
        yield SectionRule("Actions")
        with Horizontal(classes="btn-row"):
            yield Button("Init", id="init-btn", disabled=True, variant="success")
            yield Button("Curate", id="curate-btn", disabled=True, variant="success")
            yield Button("Generate", id="gen-btn", disabled=True, variant="success")
            yield Button("Prepare", id="prepare-btn", disabled=True, variant="success")
        yield SectionRule("Log")
        yield LogView(id="synth-log")

    def refresh_content(self) -> None:
        if not self.domain:
            return
        self.query_one(ConfigForm).set_domain(self.domain)
        ws = Path("workspaces") / self.domain
        status = infer_status(ws)
        self.query_one("#init-btn", Button).disabled = False
        self.query_one("#curate-btn", Button).disabled = status_order(status) < status_order(Status.EMPTY)
        self.query_one("#gen-btn", Button).disabled = status_order(status) < status_order(Status.SEEDED)
        self.query_one("#prepare-btn", Button).disabled = status_order(status) < status_order(Status.GENERATED)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if not self.domain:
            return
        bid = event.button.id
        event.stop()
        if bid == "init-btn":
            self._run_cmd(["python3", "cli.py", "init", self.domain,
                           "--desc", f"{self.domain} domain"])
        elif bid == "curate-btn":
            self._run_cmd(["python3", "cli.py", "curate", self.domain])
        elif bid == "gen-btn":
            self._run_cmd(["python3", "cli.py", "generate", self.domain])
        elif bid == "prepare-btn":
            ws = Path("workspaces") / self.domain
            self._run_cmd(["python3", "cli.py", "prepare", self.domain,
                           "--system-prompt", "You are a helpful assistant.",
                           "--out-dir", str(ws / "processed")])

    @work(thread=True)
    def _run_cmd(self, cmd: list[str]) -> None:
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
        if event.exit_code != 0:
            self.query_one(LogView).write_line(
                f"[red]Failed (exit {event.exit_code})[/red]"
            )
        self.refresh_content()
        self.call_later(self.app._rescan)
