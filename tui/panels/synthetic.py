import os
import subprocess
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Checkbox, Label, Rule
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
    DEFAULT_CSS = """
    SyntheticPanel { height: 100%; padding: 1 1 0 1; }
    SyntheticPanel #synth-config-form { height: auto; max-height: 50%; overflow-y: auto; }
    SyntheticPanel #verbose-log > .toggle--button { color: $panel; }
    SyntheticPanel #verbose-log.-on > .toggle--button { color: $text-success; }
    """

    def compose(self) -> ComposeResult:
        yield ConfigForm(_SYNTH_FIELDS, id="synth-config-form")
        yield SectionRule("Actions")
        with Horizontal(classes="btn-row"):
            yield Button("Init", id="init-btn", disabled=True, variant="success")
            yield Button("Curate", id="curate-btn", disabled=True, variant="success")
            yield Button("Generate", id="gen-btn", disabled=True, variant="success")
            yield Button("Prepare", id="prepare-btn", disabled=True, variant="success")
        yield Checkbox("Verbose log (full request/response per item)", id="verbose-log")
        yield SectionRule("Log")
        yield LogView(id="synth-log")
        yield Rule()

    def refresh_content(self) -> None:
        if not self.domain:
            return
        self.query_one(ConfigForm).set_domain(self.domain)
        ws = Path("workspaces") / self.domain
        initialized = ws.exists()
        status = infer_status(ws)
        self.query_one("#init-btn", Button).disabled = False
        self.query_one("#curate-btn", Button).disabled = not initialized
        self.query_one("#gen-btn", Button).disabled = status_order(status) < status_order(Status.SEEDED)
        self.query_one("#prepare-btn", Button).disabled = status_order(status) < status_order(Status.GENERATED)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if not self.domain:
            return
        bid = event.button.id
        event.stop()
        event.button.disabled = True
        log = self.query_one(LogView)
        if bid == "init-btn":
            log.write_line(f"Initialising domain '{self.domain}'…")
            self._run_cmd(["python3", "cli.py", "init", self.domain,
                           "--desc", f"{self.domain} domain"], finish_id=bid)
        elif bid == "curate-btn":
            log.write_line(f"Curating seeds for '{self.domain}'…")
            self._run_cmd(["python3", "cli.py", "curate", self.domain], finish_id=bid)
        elif bid == "gen-btn":
            log.write_line(f"Generating synthetic data for '{self.domain}' — this may take a while…")
            cmd = ["python3", "cli.py", "generate", self.domain]
            if self.query_one("#verbose-log", Checkbox).value:
                cmd.append("--verbose")
            self._run_cmd(cmd, finish_id=bid)
        elif bid == "prepare-btn":
            log.write_line(f"Preparing data splits for '{self.domain}'…")
            ws = Path("workspaces") / self.domain
            self._run_cmd(["python3", "cli.py", "prepare", self.domain,
                           "--system-prompt", "You are a helpful assistant.",
                           "--out-dir", str(ws / "processed")], finish_id=bid)

    @work(thread=True)
    def _run_cmd(self, cmd: list[str], finish_id: str = "") -> None:
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env,
        )
        for line in proc.stdout:
            self.post_message(RunnerOutput(line.rstrip()))
        proc.wait()
        self.post_message(RunnerDone(proc.returncode, tag=finish_id))

    def on_runner_output(self, event: RunnerOutput) -> None:
        self.query_one(LogView).write_line(event.line)

    def on_runner_done(self, event: RunnerDone) -> None:
        log = self.query_one(LogView)
        if event.exit_code != 0:
            log.write_line(f"Failed (exit {event.exit_code})")
        else:
            log.write_line("Done.")
        self.refresh_content()
        self.call_later(self.app._rescan)
