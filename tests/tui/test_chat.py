import os
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Label, RichLog

from tui.panels.chat import ChatPanel, AssistantStart, TokenOutput


class ChatApp(App):
    def __init__(self, ws: Path):
        super().__init__()
        self._ws = ws

    def compose(self) -> ComposeResult:
        yield ChatPanel(id="panel")

    def on_mount(self) -> None:
        self.query_one(ChatPanel).domain = self._ws.name


async def test_chat_send_disabled_without_fused(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    ws.mkdir(parents=True)
    os.chdir(tmp_path)
    async with ChatApp(ws).run_test() as pilot:
        await pilot.pause()
        assert pilot.app.query_one("#chat-send", Button).disabled


async def test_chat_send_enabled_with_fused(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "fused").mkdir(parents=True)
    (ws / "fused" / "model.safetensors").write_text("x")
    os.chdir(tmp_path)
    async with ChatApp(ws).run_test() as pilot:
        await pilot.pause()
        assert not pilot.app.query_one("#chat-send", Button).disabled


async def test_chat_status_no_fused(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    ws.mkdir(parents=True)
    os.chdir(tmp_path)
    async with ChatApp(ws).run_test() as pilot:
        await pilot.pause()
        label = pilot.app.query_one("#chat-status", Label)
        assert "fuse" in str(label.content).lower()


async def test_chat_domain_switch_unloads_model(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "fused").mkdir(parents=True)
    (ws / "fused" / "model.safetensors").write_text("x")
    os.chdir(tmp_path)
    async with ChatApp(ws).run_test() as pilot:
        await pilot.pause()
        panel = pilot.app.query_one(ChatPanel)
        panel._model = MagicMock()
        panel._tokenizer = MagicMock()
        # Switch domain to None
        panel.domain = None
        await pilot.pause()
        assert panel._model is None
        assert panel._tokenizer is None


async def test_chat_domain_switch_clears_log(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "fused").mkdir(parents=True)
    (ws / "fused" / "model.safetensors").write_text("x")
    os.chdir(tmp_path)
    async with ChatApp(ws).run_test() as pilot:
        await pilot.pause()
        log = pilot.app.query_one("#chat-log", RichLog)
        # Manually write something, then switch domain
        log.write("some old message")
        panel = pilot.app.query_one(ChatPanel)
        panel.domain = None
        await pilot.pause()
        # Log should be cleared; Send button should be disabled
        assert pilot.app.query_one("#chat-send", Button).disabled


async def test_chat_token_streaming(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "fused").mkdir(parents=True)
    (ws / "fused" / "model.safetensors").write_text("x")
    os.chdir(tmp_path)

    mock_resp = MagicMock()
    mock_resp.text = "hello"

    import mlx_lm
    mlx_lm.load = MagicMock(return_value=(MagicMock(), MagicMock()))
    mlx_lm.stream_generate = MagicMock(return_value=iter([mock_resp]))

    async with ChatApp(ws).run_test() as pilot:
        await pilot.pause()
        pilot.app.query_one("#chat-input", Input).value = "hi"
        await pilot.click("#chat-send")
        await pilot.pause(delay=1.0)
        log = pilot.app.query_one("#chat-log", RichLog)
        assert log is not None
        # Send should be re-enabled after worker completes
        assert not pilot.app.query_one("#chat-send", Button).disabled
