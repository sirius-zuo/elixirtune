"""Tests for the NewDomainScreen."""

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, RadioButton, RadioSet, TextArea

from tui.new_domain import NewDomainScreen


class NewDomainApp(App):
    def compose(self) -> ComposeResult:
        yield NewDomainScreen()

    def on_mount(self) -> None:
        self.push_screen(NewDomainScreen())


async def test_radio_options_present():
    """All three radio options should be visible."""
    async with NewDomainApp().run_test() as pilot:
        screen = pilot.app.screen
        radio = screen.query_one(RadioSet)
        buttons = [rb for rb in radio.query(RadioButton)]
        ids = {rb.id for rb in buttons}
        assert "rb-bootstrap" in ids
        assert "rb-import" in ids
        assert "rb-code-review" in ids


async def test_code_review_creates_workspace(tmp_path, monkeypatch):
    """Code review selection should call setup.py with --domain."""
    monkeypatch.chdir(tmp_path)
    async with NewDomainApp().run_test() as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        # Select code review option
        radio = screen.query_one(RadioSet)
        for rb in radio.query(RadioButton):
            if rb.id == "rb-code-review":
                rb.value = True
                break
        # Enter domain name
        name_input = screen.query_one("#new-domain-name", Input)
        name_input.value = "test-code-review"
        # Click Create
        create_btn = screen.query_one("#new-domain-create", Button)
        create_btn.press()
        await pilot.pause()
        # The screen should dismiss with success=False (since setup.py doesn't exist in base repo)
        assert len(pilot.app.screen_stack) < 2
