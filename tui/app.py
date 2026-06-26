from pathlib import Path

from textual.app import App, ComposeResult
from textual.css.query import QueryType
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
    def __init__(
        self,
        initial_domain: str | None = None,
        root: Path = Path("."),
    ) -> None:
        super().__init__()
        self._initial_domain = initial_domain
        self._root = Path(root)

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

    def on_mount(self) -> None:
        self._rescan()
        target = self._initial_domain
        if not target:
            domains = scan_domains(self._root)
            target = domains[0].name if domains else None
        if target:
            self._switch_domain(target)

    def on_domain_selected(self, event: DomainSelected) -> None:
        self._switch_domain(event.domain)

    def on_new_domain_requested(self, _: NewDomainRequested) -> None:
        from tui.new_domain import NewDomainScreen
        self.push_screen(NewDomainScreen(root=self._root))

    def query_one(
        self,
        selector: str | type[QueryType],
        expect_type: type[QueryType] | None = None,
    ) -> QueryType | Widget:
        """Query for a widget, including widgets in modal screens."""
        try:
            # Try the default query first (which searches the current screen stack)
            return super().query_one(selector, expect_type)
        except Exception:
            # If not found in the primary path, search all screens in the stack
            if isinstance(selector, str) and len(self.screen_stack) > 1:
                # Try querying each screen in reverse order (top to bottom)
                for screen in reversed(self.screen_stack):
                    try:
                        return screen.query_one(selector, expect_type)
                    except Exception:
                        continue
            raise

    def _switch_domain(self, domain: str) -> None:
        for panel in self.query(BasePanel):
            panel.domain = domain

    def _rescan(self) -> None:
        domains = scan_domains(self._root)
        self.query_one(Sidebar).refresh_domains(domains)
