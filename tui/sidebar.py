from textual.app import ComposeResult
from textual.message import Message
from textual.widget import Widget
from textual.widgets import Button, Label, ListView, ListItem
from textual.events import Click

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


class DomainLabel(Label):
    """A Label that can post DomainSelected when clicked."""

    def __init__(self, text: str, domain_name: str, **kwargs):
        super().__init__(text, **kwargs)
        self.domain_name = domain_name

    def on_click(self, event: Click) -> None:
        """Handle click and post DomainSelected message."""
        # Find the Sidebar ancestor and post the message
        parent = self.parent
        while parent:
            if isinstance(parent, Sidebar):
                parent.post_message(DomainSelected(self.domain_name))
                break
            parent = parent.parent


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

    def refresh_domains(self, domains: list[DomainState]) -> None:
        lv = self.query_one(ListView)
        lv.clear()
        for d in domains:
            dot = _DOT[d.status]
            label = DomainLabel(f"{dot} {d.name}", domain_name=d.name)
            lv.append(ListItem(label, id=f"domain-{d.name}"))

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item.id and event.item.id.startswith("domain-"):
            self.post_message(DomainSelected(event.item.id[len("domain-"):]))

    def on_click(self, event: Click) -> None:
        """Handle click on list items and select them."""
        # Get the widget that was clicked
        widget = event.widget

        # Walk up to find the ListItem
        listitem = widget
        while listitem and not isinstance(listitem, ListItem):
            listitem = listitem.parent

        # If we found a ListItem, treat it as selected
        if listitem and isinstance(listitem, ListItem) and listitem.id:
            if listitem.id.startswith("domain-"):
                domain_name = listitem.id[len("domain-"):]
                self.post_message(DomainSelected(domain_name))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new-domain-btn":
            self.post_message(NewDomainRequested())
