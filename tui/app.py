from pathlib import Path

from textual.app import App, ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Header, TabbedContent, TabPane, Label

from tui.domain import scan_domains, read_domain_type
from tui.sidebar import Sidebar, DomainSelected, NewDomainRequested, DeleteDomainRequested


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
from tui.panels.embedding_training import EmbeddingTrainingPanel
from tui.panels.embedding_eval import EmbeddingEvalPanel


_LM_TABS = ["tab-overview", "tab-synth", "tab-training", "tab-eval", "tab-deploy", "tab-chat"]
_EMBED_TABS = ["tab-overview", "tab-embed-data", "tab-embed-train", "tab-embed-eval"]
_ALL_TABS = list(dict.fromkeys(_LM_TABS + _EMBED_TABS))  # ordered, deduped


class ElixirTuneApp(App):
    TITLE = "ElixirTune"
    CSS = """
    #main-tabs { height: 1fr; }
    TabPane { height: 1fr; }
    Button.-success { background: #0178D4; color: #ffffff; }
    Button.-success:hover { background: #3399e0; color: #ffffff; }
    Button.-success:focus { background: #0178D4; color: #ffffff; }
    Button.-success:disabled { background: #1a4a6b; color: #555555; }
    .btn-row { height: auto; }
    .btn-row Button { margin-right: 2; }
    Rule { color: #0178D4; margin: 0; }
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
            # LM-only tabs
            with TabPane("Synth", id="tab-synth"):
                yield SyntheticPanel(id="panel-synth")
            with TabPane("Training", id="tab-training"):
                yield TrainingPanel(id="panel-training")
            with TabPane("Eval", id="tab-eval"):
                yield EvaluationPanel(id="panel-eval")
            with TabPane("Deploy", id="tab-deploy"):
                yield DeploymentPanel(id="panel-deploy")
            with TabPane("Chat", id="tab-chat"):
                from tui.panels.chat import ChatPanel
                yield ChatPanel(id="panel-chat")
            # Embedding-only tabs
            with TabPane("Embed Data", id="tab-embed-data"):
                yield Label("Import or convert data using the buttons below.")
            with TabPane("Embed Train", id="tab-embed-train"):
                yield EmbeddingTrainingPanel(id="panel-embed-train")
            with TabPane("Embed Eval", id="tab-embed-eval"):
                yield EmbeddingEvalPanel(id="panel-embed-eval")

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

    def on_delete_domain_requested(self, _: DeleteDomainRequested) -> None:
        from tui.delete_domain import DeleteDomainScreen
        domains = [d.name for d in scan_domains(self._root)]
        if not domains:
            self.notify("No domains to delete.", severity="warning")
            return

        def _on_deleted(result: dict | None) -> None:
            if not result:
                return
            deleted = result["deleted"]
            self.notify(f"Domain '{deleted}' deleted.")
            if self._current_domain == deleted:
                self._current_domain = None
                for panel in self.query(BasePanel):
                    panel.domain = None
            self.call_later(self._rescan)

        self.push_screen(DeleteDomainScreen(domains=domains, root=self._root), _on_deleted)

    def on_new_domain_requested(self, _: NewDomainRequested) -> None:
        from tui.new_domain import NewDomainScreen

        def _on_created(result: dict | None) -> None:
            if not result:
                return
            if result.get("success"):
                self._switch_domain(result["name"])
            else:
                self.notify(result.get("error", "Failed to create domain."), severity="error")

        self.push_screen(NewDomainScreen(root=self._root), _on_created)

    def on_tabbed_content_tab_activated(self, event: TabbedContent.TabActivated) -> None:
        for panel in event.pane.query(BasePanel):
            panel.refresh_content()

    def _switch_domain(self, domain: str) -> None:
        self._current_domain = domain
        for panel in self.query(BasePanel):
            panel.domain = domain
        ws = self._root / "workspaces" / domain
        domain_type = read_domain_type(ws)
        self._update_tabs_for_type(domain_type)
        self.call_later(self._rescan)

    def _update_tabs_for_type(self, domain_type: str) -> None:
        tc = self.query_one(TabbedContent)
        visible = set(_LM_TABS if domain_type == "lm" else _EMBED_TABS)
        for tab_id in _ALL_TABS:
            if tab_id == "tab-overview":
                continue  # always visible
            try:
                if tab_id in visible:
                    tc.show_tab(tab_id)
                else:
                    tc.hide_tab(tab_id)
            except Exception:
                pass

    async def _rescan(self) -> None:
        domains = scan_domains(self._root)
        await self.query_one(Sidebar).refresh_domains(
            domains, active=self._current_domain
        )
