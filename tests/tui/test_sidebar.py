import pytest
import yaml
from pathlib import Path
from textual.app import App, ComposeResult
from tui.sidebar import Sidebar, DomainSelected, NewDomainRequested
from tui.domain import DomainState, Status


def _make_domains():
    return [
        DomainState("alpha", Path("workspaces/alpha"), Status.SEEDED),
        DomainState("beta", Path("workspaces/beta"), Status.TRAINED),
    ]


class SidebarApp(App):
    def compose(self) -> ComposeResult:
        yield Sidebar(id="sidebar")

    async def on_mount(self) -> None:
        await self.query_one(Sidebar).refresh_domains(_make_domains())


async def test_sidebar_renders_domain_names():
    async with SidebarApp().run_test() as pilot:
        text = pilot.app.query_one(Sidebar).renderable if hasattr(pilot.app.query_one(Sidebar), "renderable") else ""
        # Check list items exist
        from textual.widgets import ListItem
        items = pilot.app.query(ListItem)
        labels = [str(i.id) for i in items]
        assert "domain-alpha" in labels
        assert "domain-beta" in labels


async def test_sidebar_click_emits_domain_selected():
    received = []

    class App2(SidebarApp):
        def on_domain_selected(self, e: DomainSelected):
            received.append(e.domain)

    async with App2().run_test() as pilot:
        await pilot.pause()
        await pilot.click("#domain-alpha")
        await pilot.pause()
        assert "alpha" in received


async def test_sidebar_new_domain_button_emits_message():
    received = []

    class App3(SidebarApp):
        def on_new_domain_requested(self, e: NewDomainRequested):
            received.append(True)

    async with App3().run_test() as pilot:
        await pilot.click("#new-domain-btn")
        await pilot.pause()
        assert received


async def test_sidebar_shows_em_badge_for_embedding_domain(tmp_path):
    ws = tmp_path / "workspaces" / "emb"
    ws.mkdir(parents=True)
    (ws / "config.yaml").write_text(yaml.safe_dump({"type": "embedding"}))
    from tui.app import ElixirTuneApp
    async with ElixirTuneApp(root=tmp_path).run_test() as pilot:
        from textual.widgets import ListItem, Label
        items = list(pilot.app.query(ListItem))
        for item in items:
            label = item.query_one(Label, Label)
            assert "[EM]" in label.content
