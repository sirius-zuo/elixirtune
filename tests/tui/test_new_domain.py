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
        radio = screen.query_one("#source-radio", RadioSet)
        ids = {rb.id for rb in radio.query(RadioButton)}
        assert "rb-bootstrap" in ids
        assert "rb-import" in ids
        assert "rb-code-review" not in ids


async def test_new_domain_has_type_selector():
    """The new domain dialog should have LM and Embedding type radio buttons."""
    async with NewDomainApp().run_test() as pilot:
        screen = pilot.app.screen
        type_radio = screen.query_one("#type-radio", RadioSet)
        type_ids = {rb.id for rb in type_radio.query(RadioButton)}
        assert "rb-type-lm" in type_ids
        assert "rb-type-embedding" in type_ids
