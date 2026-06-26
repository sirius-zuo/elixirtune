from pathlib import Path

import yaml
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual import work
from textual.widgets import Button, Input, Label, RichLog, Rule

from tui.app import BasePanel
from tui.runner import RunnerDone


class AssistantStart(Message):
    pass


class TokenOutput(Message):
    def __init__(self, token: str) -> None:
        super().__init__()
        self.token = token


class ChatPanel(BasePanel):
    DEFAULT_CSS = """
    ChatPanel { height: 100%; padding: 1; }
    #chat-log { height: 1fr; border: solid $surface; }
    #chat-input-row { height: 3; }
    #chat-input-row Input { width: 1fr; }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._model = None
        self._tokenizer = None

    def compose(self) -> ComposeResult:
        yield Label("No fused model — run Deployment › Fuse first", id="chat-status")
        yield Rule()
        yield RichLog(id="chat-log", auto_scroll=True, markup=True)
        with Horizontal(id="chat-input-row"):
            yield Input(id="chat-input", placeholder="Type a message…")
            yield Button("Send", id="chat-send", disabled=True)

    def watch_domain(self, domain: str | None) -> None:
        # Always call refresh_content, even when domain is None
        self.refresh_content()

    def refresh_content(self) -> None:
        self._unload_model()
        self.query_one("#chat-log", RichLog).clear()
        if not self.domain:
            self.query_one("#chat-status", Label).update(
                "No fused model — run Deployment › Fuse first"
            )
            self.query_one("#chat-send", Button).disabled = True
            return
        ws = Path("workspaces") / self.domain
        fused = ws / "fused"
        if fused.exists() and any(fused.iterdir()):
            self.query_one("#chat-status", Label).update(
                "Ready (model not loaded — send a message to load)"
            )
            self.query_one("#chat-send", Button).disabled = False
        else:
            self.query_one("#chat-status", Label).update(
                "No fused model — run Deployment › Fuse first"
            )
            self.query_one("#chat-send", Button).disabled = True

    def _unload_model(self) -> None:
        self._model = None
        self._tokenizer = None

    def _system_prompt(self, domain: str) -> str:
        cfg_path = Path("workspaces") / domain / "config.yaml"
        if cfg_path.exists():
            try:
                data = yaml.safe_load(cfg_path.read_text()) or {}
                if isinstance(data, dict):
                    sp = data.get("chat", {}).get("system_prompt")
                    if sp:
                        return sp
            except Exception:
                pass
        return "You are a helpful assistant."

    def _format_prompt(self, message: str, domain: str) -> str:
        system = self._system_prompt(domain)
        return f"<|system|>\n{system}<|end|>\n<|user|>\n{message}<|end|>\n<|assistant|>"

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "chat-send":
            self._send()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "chat-input":
            self._send()

    def _send(self) -> None:
        if self.query_one("#chat-send", Button).disabled:
            return
        inp = self.query_one("#chat-input", Input)
        message = inp.value.strip()
        if not message:
            return
        inp.clear()
        self.query_one("#chat-log", RichLog).write(
            f"[bold cyan]You:[/bold cyan] {message}\n"
        )
        self.query_one("#chat-send", Button).disabled = True
        status = "Loading model…" if self._model is None else "Generating…"
        self.query_one("#chat-status", Label).update(status)
        self._send_message(message, self.domain)

    @work(thread=True)
    def _send_message(self, message: str, domain: str) -> None:
        import mlx_lm
        ws = Path("workspaces") / domain
        if self._model is None:
            try:
                self._model, self._tokenizer = mlx_lm.load(str(ws / "fused"))
            except Exception as e:
                self.post_message(TokenOutput(f"\n[red]Failed to load model: {e}[/red]"))
                self.post_message(RunnerDone(1))
                return
        prompt = self._format_prompt(message, domain)
        self.post_message(AssistantStart())
        exit_code = 0
        try:
            for response in mlx_lm.stream_generate(
                self._model, self._tokenizer, prompt, max_tokens=512
            ):
                self.post_message(TokenOutput(response.text))
        except Exception as e:
            self.post_message(TokenOutput(f"\n[red]Error: {e}[/red]"))
            exit_code = 1
        self.post_message(RunnerDone(exit_code))

    def on_assistant_start(self, _: AssistantStart) -> None:
        self.query_one("#chat-log", RichLog).write(
            "[bold green]Assistant:[/bold green] "
        )
        self.query_one("#chat-status", Label).update("Generating…")

    def on_token_output(self, event: TokenOutput) -> None:
        self.query_one("#chat-log", RichLog).write(event.token)

    def on_runner_done(self, event: RunnerDone) -> None:
        self.query_one("#chat-log", RichLog).write("\n")
        if event.exit_code != 0:
            self.query_one("#chat-status", Label).update("Error — see log above")
        else:
            # Re-enable Send only if domain is still valid and has a fused model
            if self.domain:
                ws = Path("workspaces") / self.domain
                fused = ws / "fused"
                if fused.exists() and any(fused.iterdir()):
                    self.query_one("#chat-send", Button).disabled = False
                    self.query_one("#chat-status", Label).update("Ready")
                    return
            # Domain changed or no fused model — keep Send disabled
            self.query_one("#chat-status", Label).update(
                "No fused model — run Deployment › Fuse first"
            )
