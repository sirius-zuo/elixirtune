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
    """The two source options should be visible (code-review option was removed)."""
    async with NewDomainApp().run_test() as pilot:
        screen = pilot.app.screen
        radio = screen.query_one(RadioSet)
        ids = {rb.id for rb in radio.query(RadioButton)}
        assert "rb-bootstrap" in ids
        assert "rb-import" in ids
        assert "rb-code-review" not in ids
