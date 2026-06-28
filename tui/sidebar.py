from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Label, ListView, ListItem
from tui.domain import DomainState, Status

_DOT = {
    Status.DEPLOYED: "●", Status.EVALUATED: "●",
    Status.TRAINED: "◉", Status.PREPARED: "◉", Status.GENERATED: "◉",
    Status.SEEDED: "○", Status.EMPTY: "○",
}


class DomainSelected(Message):
    def __init__(self, domain: str) -> None:
        super().__init__()
        self.domain = domain


class NewDomainRequested(Message):
    pass


class Sidebar(Widget):
    DEFAULT_CSS = """
    Sidebar { width: 22; dock: left; border-right: solid $primary; }
    #sidebar-title { text-style: bold; padding: 1 1 0 1; }
    #domain-list { height: 1fr; }
    #new-domain-btn { dock: bottom; width: 100%; }
    """

    def compose(self) -> ComposeResult:
        yield Label("DOMAINS", id="sidebar-title")
        yield ListView(id="domain-list")
        yield Button("+ New Domain", id="new-domain-btn")

    async def refresh_domains(self, domains: list[DomainState], active: str | None = None) -> None:
        lv = self.query_one(ListView)
        await lv.clear()
        active_index = None
        for i, d in enumerate(domains):
            dot = "●" if d.name == active else _DOT[d.status]
            lv.append(ListItem(Label(f"{dot} {d.name}"), id=f"domain-{d.name}"))
            if d.name == active:
                active_index = i
        if active_index is not None:
            # Use call_after_refresh so items are fully mounted before setting index
            self.call_after_refresh(lambda idx=active_index: setattr(lv, "index", idx))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item.id and event.item.id.startswith("domain-"):
            self.post_message(DomainSelected(event.item.id[len("domain-"):]))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new-domain-btn":
            self.post_message(NewDomainRequested())
