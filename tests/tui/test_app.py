import pytest
from pathlib import Path
from tui.app import ElixirLoRAApp
from tui.sidebar import Sidebar
from textual.widgets import TabbedContent


async def test_app_mounts_sidebar_and_tabs(tmp_path):
    async with ElixirLoRAApp(root=tmp_path).run_test() as pilot:
        assert pilot.app.query_one(Sidebar) is not None
        assert pilot.app.query_one(TabbedContent) is not None


async def test_app_loads_domains_into_sidebar(tmp_path):
    (tmp_path / "workspaces" / "mydom").mkdir(parents=True)
    async with ElixirLoRAApp(root=tmp_path).run_test() as pilot:
        from textual.widgets import ListItem
        ids = [i.id for i in pilot.app.query(ListItem)]
        assert "domain-mydom" in ids


async def test_app_switching_domain_sets_panel_domain(tmp_path):
    (tmp_path / "workspaces" / "d1").mkdir(parents=True)
    async with ElixirLoRAApp(initial_domain="d1", root=tmp_path).run_test() as pilot:
        from tui.app import BasePanel
        panels = list(pilot.app.query(BasePanel))
        assert all(p.domain == "d1" for p in panels)
