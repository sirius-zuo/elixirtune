from pathlib import Path

import yaml
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.message import Message
from textual import work
from textual.widgets import Button, Input, Label, TextArea

from tui.app import BasePanel
from tui.runner import RunnerDone
from tui.widgets.section_rule import SectionRule


_STOP_STRINGS = {"<|im_end|>", "<|endoftext|>", "<|end|>", "<|eot_id|>"}


def _strip_stop_tokens(text: str) -> tuple[str, bool]:
    """Return (cleaned_text, hit_stop). Strips trailing stop sequences."""
    for stop in _STOP_STRINGS:
        idx = text.find(stop)
        if idx != -1:
            return text[:idx], True
    return text, False


class AssistantStart(Message):
    pass


class TokenOutput(Message):
    def __init__(self, token: str) -> None:
        super().__init__()
        self.token = token


class ChatPanel(BasePanel):
    DEFAULT_CSS = """
    ChatPanel { height: 100%; padding: 1; }
    #chat-log { height: 1fr; border: solid #0178D4; }
    #chat-input-row { height: 3; }
    #chat-input-row Input { width: 1fr; }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._model = None
        self._tokenizer = None
        self._chat_text = ""
        self._generating = False

    def compose(self) -> ComposeResult:
        yield SectionRule("Status")
        yield Label("No fused model — run Deployment › Fuse first", id="chat-status")
        yield SectionRule("Conversation")
        yield TextArea("", id="chat-log", read_only=True)
        with Horizontal(id="chat-input-row"):
            yield Input(id="chat-input", placeholder="Type a message…")
            yield Button("Send", id="chat-send", disabled=True, variant="success")

    def watch_domain(self, domain: str | None) -> None:
        self.refresh_content()

    def refresh_content(self) -> None:
        self._unload_model()
        self._chat_text = ""
        self.query_one("#chat-log", TextArea).load_text("")
        if not self.domain:
            self.query_one("#chat-status", Label).update(
                "No fused model — run Deployment › Fuse first"
            )
            self.query_one("#chat-send", Button).disabled = True
            return
        ws = Path("workspaces") / self.domain
        fused = ws / "fused"
        if fused.exists() and any(fused.iterdir()):
            self.query_one("#chat-status", Label).update("Ready")
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
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": message},
        ]
        try:
            return self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            # Fallback to ChatML if apply_chat_template fails
            return (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n{message}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

    def _append_chat(self, text: str) -> None:
        self._chat_text += text
        ta = self.query_one("#chat-log", TextArea)
        ta.load_text(self._chat_text)
        ta.scroll_end(animate=False)

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
        self._append_chat(f"You: {message}\n")
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
                self.post_message(TokenOutput(f"\nFailed to load model: {e}\n"))
                self.post_message(RunnerDone(1))
                return
        prompt = self._format_prompt(message, domain)
        self.post_message(AssistantStart())
        exit_code = 0
        try:
            for response in mlx_lm.stream_generate(
                self._model, self._tokenizer, prompt, max_tokens=512
            ):
                text, stopped = _strip_stop_tokens(response.text)
                if text:
                    self.post_message(TokenOutput(text))
                if stopped:
                    break
        except Exception as e:
            self.post_message(TokenOutput(f"\nError: {e}\n"))
            exit_code = 1
        self.post_message(RunnerDone(exit_code))

    def on_assistant_start(self, _: AssistantStart) -> None:
        self._append_chat("Assistant: ")
        self.query_one("#chat-status", Label).update("Generating…")

    def on_token_output(self, event: TokenOutput) -> None:
        self._append_chat(event.token)

    def on_runner_done(self, event: RunnerDone) -> None:
        self._append_chat("\n\n")
        if event.exit_code != 0:
            self.query_one("#chat-status", Label).update("Error — see log above")
        else:
            if self.domain:
                ws = Path("workspaces") / self.domain
                fused = ws / "fused"
                if fused.exists() and any(fused.iterdir()):
                    self.query_one("#chat-send", Button).disabled = False
                    self.query_one("#chat-status", Label).update("Ready")
                    return
            self.query_one("#chat-status", Label).update(
                "No fused model — run Deployment › Fuse first"
            )
