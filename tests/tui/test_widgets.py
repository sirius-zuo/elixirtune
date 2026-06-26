import pytest
import yaml
from textual.app import App, ComposeResult
from tui.widgets.log_view import LogView
from tui.widgets.config_form import ConfigField, ConfigForm


class LogApp(App):
    def compose(self) -> ComposeResult:
        yield LogView(id="log")


async def test_log_view_write_and_clear():
    async with LogApp().run_test() as pilot:
        lv = pilot.app.query_one(LogView)
        lv.write_line("hello")
        await pilot.pause()
        lv.clear()
        await pilot.pause()
        # just assert no exception and widget is mounted
        assert pilot.app.query_one(LogView) is not None


class CfgApp(App):
    def __init__(self, fields, domain=""):
        super().__init__()
        self._fields = fields
        self._domain = domain

    def compose(self) -> ComposeResult:
        yield ConfigForm(self._fields, domain=self._domain, id="form")


async def test_config_form_loads_existing_value(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "workspaces" / "d" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(yaml.safe_dump({"teacher": {"model": "qwen3.6"}}))

    fields = [ConfigField("Model", "workspaces/{domain}/config.yaml", ["teacher", "model"])]
    async with CfgApp(fields, domain="d").run_test() as pilot:
        inp = pilot.app.query_one("#cfg-model")
        assert inp.value == "qwen3.6"


async def test_config_form_save_writes_yaml(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = tmp_path / "workspaces" / "d" / "config.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(yaml.safe_dump({"teacher": {"model": "old"}}))

    fields = [ConfigField("Model", "workspaces/{domain}/config.yaml", ["teacher", "model"])]

    async with CfgApp(fields, domain="d").run_test() as pilot:
        pilot.app.query_one("#cfg-model").value = "new-model"
        saved_events = []
        pilot.app.query_one(ConfigForm).on(ConfigForm.Saved, lambda e: saved_events.append(e))
        await pilot.click("#cfg-save")
        await pilot.pause()

    written = yaml.safe_load(cfg.read_text())
    assert written["teacher"]["model"] == "new-model"
