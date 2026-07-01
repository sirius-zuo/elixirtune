import os
import sys
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from textual.app import App, ComposeResult
from textual.widgets import Button, Checkbox, Input, Label
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tui.upload_modal import HFUploadScreen
import cli

runner = CliRunner()


class ModalApp(App):
    def compose(self) -> ComposeResult:
        return iter([])

    def on_mount(self) -> None:
        self.push_screen(HFUploadScreen())


async def test_upload_modal_cancel_dismisses():
    async with ModalApp().run_test() as pilot:
        await pilot.click("#hf-cancel-btn")
        await pilot.pause()
        assert not any(isinstance(s, HFUploadScreen) for s in pilot.app.screen_stack)


async def test_upload_modal_validation_empty_repo():
    async with ModalApp().run_test() as pilot:
        pilot.app.screen.query_one("#hf-token", Input).value = "tok"
        await pilot.click("#hf-upload-confirm")
        await pilot.pause()
        label = pilot.app.screen.query_one("#hf-error", Label)
        assert "required" in str(label.content).lower()


async def test_upload_modal_validation_no_slash():
    async with ModalApp().run_test() as pilot:
        pilot.app.screen.query_one("#hf-repo-name", Input).value = "badrepo"
        pilot.app.screen.query_one("#hf-token", Input).value = "tok"
        await pilot.click("#hf-upload-confirm")
        await pilot.pause()
        label = pilot.app.screen.query_one("#hf-error", Label)
        assert "username/repo" in str(label.content).lower()


async def test_upload_modal_validation_empty_token():
    async with ModalApp().run_test() as pilot:
        pilot.app.screen.query_one("#hf-repo-name", Input).value = "user/repo"
        pilot.app.screen.query_one("#hf-token", Input).value = ""
        await pilot.click("#hf-upload-confirm")
        await pilot.pause()
        label = pilot.app.screen.query_one("#hf-error", Label)
        assert "token" in str(label.content).lower()


def test_upload_cli_missing_fused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(cli.app, ["upload", "d", "--repo-name", "u/r", "--token", "tok"])
    assert result.exit_code == 1
    assert "fused" in result.output.lower()


def test_upload_cli_missing_token(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ws = tmp_path / "workspaces" / "d" / "fused"
    ws.mkdir(parents=True)
    (ws / "model.safetensors").write_text("x")
    monkeypatch.delenv("HF_TOKEN", raising=False)
    result = runner.invoke(cli.app, ["upload", "d", "--repo-name", "u/r"])
    assert result.exit_code == 1
    assert "token" in result.output.lower()


def test_upload_cli_success(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ws = tmp_path / "workspaces" / "d" / "fused"
    ws.mkdir(parents=True)
    (ws / "model.safetensors").write_text("x")
    with patch("huggingface_hub.create_repo") as mock_create, \
         patch("huggingface_hub.upload_folder") as mock_upload:
        result = runner.invoke(cli.app, ["upload", "d", "--repo-name", "u/r", "--token", "tok"])
    assert result.exit_code == 0
    mock_create.assert_called_once()
    mock_upload.assert_called_once()
    assert "u/r" in result.output


async def test_upload_modal_validation_empty_side_of_slash():
    async with ModalApp().run_test() as pilot:
        pilot.app.screen.query_one("#hf-repo-name", Input).value = "user/"
        pilot.app.screen.query_one("#hf-token", Input).value = "tok"
        await pilot.click("#hf-upload-confirm")
        await pilot.pause()
        label = pilot.app.screen.query_one("#hf-error", Label)
        assert "username/repo" in str(label.content).lower()


async def test_upload_modal_labels_have_no_border():
    """Labels must not be bordered, or their 1-row height leaves no room for text."""
    async with ModalApp().run_test(size=(80, 30)) as pilot:
        for label in pilot.app.screen.query(Label):
            assert label.styles.border_top[0] == ""


async def test_upload_modal_fields_fit_within_dialog_width():
    """Input/Checkbox must be constrained to the dialog width, not stretched full-screen."""
    async with ModalApp().run_test(size=(80, 30)) as pilot:
        repo_width = pilot.app.screen.query_one("#hf-repo-name", Input).region.width
        token_width = pilot.app.screen.query_one("#hf-token", Input).region.width
        assert repo_width <= 60
        assert token_width <= 60
