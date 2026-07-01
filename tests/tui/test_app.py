import pytest
import yaml
from pathlib import Path
from tui.app import ElixirTuneApp
from tui.sidebar import Sidebar
from textual.widgets import TabbedContent


async def test_app_mounts_sidebar_and_tabs(tmp_path):
    async with ElixirTuneApp(root=tmp_path).run_test() as pilot:
        assert pilot.app.query_one(Sidebar) is not None
        assert pilot.app.query_one(TabbedContent) is not None


async def test_app_loads_domains_into_sidebar(tmp_path):
    (tmp_path / "workspaces" / "mydom").mkdir(parents=True)
    async with ElixirTuneApp(root=tmp_path).run_test() as pilot:
        from textual.widgets import ListItem
        ids = [i.id for i in pilot.app.query(ListItem)]
        assert "domain-mydom" in ids


async def test_app_switching_domain_sets_panel_domain(tmp_path):
    (tmp_path / "workspaces" / "d1").mkdir(parents=True)
    async with ElixirTuneApp(initial_domain="d1", root=tmp_path).run_test() as pilot:
        from tui.app import BasePanel
        panels = list(pilot.app.query(BasePanel))
        assert all(p.domain == "d1" for p in panels)


async def test_new_domain_button_opens_modal(tmp_path):
    async with ElixirTuneApp(root=tmp_path).run_test() as pilot:
        await pilot.click("#new-domain-btn")
        await pilot.pause()
        from textual.widgets import Input
        # Modal is on screen — domain name input should be visible
        assert pilot.app.screen.query_one("#new-domain-name", Input) is not None


async def test_new_domain_cancel_closes_modal(tmp_path):
    async with ElixirTuneApp(root=tmp_path).run_test() as pilot:
        await pilot.click("#new-domain-btn")
        await pilot.pause()
        await pilot.click("#new-domain-cancel")
        await pilot.pause()
        from tui.new_domain import NewDomainScreen
        assert not any(isinstance(s, NewDomainScreen) for s in pilot.app.screen_stack)


async def test_embedding_domain_shows_embedding_tabs(tmp_path):
    ws = tmp_path / "workspaces" / "emb"
    ws.mkdir(parents=True)
    (ws / "config.yaml").write_text(yaml.safe_dump({"type": "embedding"}))
    async with ElixirTuneApp(initial_domain="emb", root=tmp_path).run_test() as pilot:
        tc = pilot.app.query_one(TabbedContent)
        assert tc.query_one("#tab-embed-train") is not None


async def test_lm_domain_shows_lm_tabs(tmp_path):
    ws = tmp_path / "workspaces" / "lm"
    ws.mkdir(parents=True)
    async with ElixirTuneApp(initial_domain="lm", root=tmp_path).run_test() as pilot:
        tc = pilot.app.query_one(TabbedContent)
        assert tc.query_one("#tab-training") is not None
