from pathlib import Path

from textual.app import App, ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Header, TabbedContent, TabPane

from tui.domain import scan_domains
from tui.sidebar import Sidebar, DomainSelected, NewDomainRequested


class BasePanel(Widget):
    domain: reactive[str | None] = reactive(None)

    def watch_domain(self, domain: str | None) -> None:
        if domain:
            self.refresh_content()

    def refresh_content(self) -> None:
        pass


# Panels are imported after BasePanel to avoid circular import issues
from tui.panels.overview import OverviewPanel
from tui.panels.synthetic import SyntheticPanel
from tui.panels.training import TrainingPanel
from tui.panels.evaluation import EvaluationPanel
from tui.panels.deployment import DeploymentPanel


class ElixirLoRAApp(App):
    TITLE = "ElixirLoRA"
    CSS = """
    #main-tabs { height: 1fr; }
    TabPane { height: 1fr; }
    Button.-success { background: #0178D4; color: #ffffff; }
    Button.-success:hover { background: #3399e0; color: #ffffff; }
    Button.-success:focus { background: #0178D4; color: #ffffff; }
    Button.-success:disabled { background: #1a4a6b; color: #555555; }
    .btn-row { height: auto; }
    .btn-row Button { margin-right: 2; }
    """

    def __init__(
        self,
        initial_domain: str | None = None,
        root: Path = Path("."),
    ) -> None:
        super().__init__()
        self._initial_domain = initial_domain
        self._root = Path(root)
        self._current_domain: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Sidebar(id="sidebar")
        with TabbedContent(id="main-tabs"):
            with TabPane("Overview", id="tab-overview"):
                yield OverviewPanel(id="panel-overview")
            with TabPane("Synth", id="tab-synth"):
                yield SyntheticPanel(id="panel-synth")
            with TabPane("Training", id="tab-training"):
                yield TrainingPanel(id="panel-training")
            with TabPane("Eval", id="tab-eval"):
                yield EvaluationPanel(id="panel-eval")
            with TabPane("Deploy", id="tab-deploy"):
                yield DeploymentPanel(id="panel-deploy")
            with TabPane("Chat", id="tab-chat"):
                from tui.panels.chat import ChatPanel  # local import avoids circular dep
                yield ChatPanel(id="panel-chat")

    async def on_mount(self) -> None:
        await self._rescan()
        target = self._initial_domain
        if not target:
            domains = scan_domains(self._root)
            target = domains[0].name if domains else None
        if target:
            self._switch_domain(target)

    def on_domain_selected(self, event: DomainSelected) -> None:
        if event.domain != self._current_domain:
            self._switch_domain(event.domain)

    def on_new_domain_requested(self, _: NewDomainRequested) -> None:
        from tui.new_domain import NewDomainScreen
        self.push_screen(NewDomainScreen(root=self._root))

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        for panel in event.pane.query(BasePanel):
            panel.refresh_content()

    def _switch_domain(self, domain: str) -> None:
        self._current_domain = domain
        for panel in self.query(BasePanel):
            panel.domain = domain
        self.call_later(self._rescan)

    async def _rescan(self) -> None:
        domains = scan_domains(self._root)
        await self.query_one(Sidebar).refresh_domains(domains, active=self._current_domain)
