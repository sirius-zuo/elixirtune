from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Input, Label, Rule, Select
from textual import work

from tui.app import BasePanel
from tui.domain import infer_status, Status
from tui.widgets.log_view import LogView
from tui.widgets.section_rule import SectionRule


class EmbeddingEvalPanel(BasePanel):
    DEFAULT_CSS = """
    EmbeddingEvalPanel { height: 100%; padding: 1 1 0 1; }
    EmbeddingEvalPanel #anchor-input { width: 1fr; }
    EmbeddingEvalPanel #candidate-input { width: 1fr; }
    EmbeddingEvalPanel #beir-select { width: 28; }
    """

    _BEIR_DATASETS = [
        ("SciFact", "scifact"),
        ("NFCorpus", "nfcorpus"),
        ("TREC-COVID", "trec-covid"),
        ("FiQA", "fiqa"),
        ("MSMARCO (dev)", "msmarco"),
    ]

    def compose(self) -> ComposeResult:
        yield SectionRule("Cosine Similarity Probe")
        yield Label("Anchor text:")
        yield Input(id="anchor-input", placeholder="Enter query or anchor text")
        yield Label("Candidates (one per line):")
        yield Input(id="candidate-input", placeholder="doc1\ndoc2\ndoc3")
        with Horizontal(classes="btn-row"):
            yield Button("Compute similarity", id="similarity-btn", disabled=True, variant="success")
        yield SectionRule("Retrieval Metrics (val set)")
        with Horizontal(classes="btn-row"):
            yield Button("Recall@1/5/10", id="recall-btn", disabled=True, variant="success")
        yield SectionRule("BEIR Benchmark")
        yield Select(
            [(label, val) for label, val in self._BEIR_DATASETS],
            value="scifact",
            allow_blank=False,
            id="beir-select",
        )
        with Horizontal(classes="btn-row"):
            yield Button("Run BEIR", id="beir-btn", disabled=True, variant="success")
        yield SectionRule("Results")
        yield Label("", id="eval-results")
        yield LogView(id="eval-log")
        yield Rule()

    def refresh_content(self) -> None:
        if not self.domain:
            return
        ws = Path("workspaces") / self.domain
        status = infer_status(ws)
        trained = status in (Status.TRAINED, Status.CE_TRAINED)
        val_ready = (ws / "processed" / "embedding_val.json").exists()

        self.query_one("#similarity-btn", Button).disabled = not trained
        self.query_one("#recall-btn", Button).disabled = not (trained and val_ready)
        self.query_one("#beir-btn", Button).disabled = not trained

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if not self.domain:
            return
        btn_id = event.button.id
        if btn_id == "similarity-btn":
            event.stop()
            anchor = self.query_one("#anchor-input", Input).value.strip()
            candidates_raw = self.query_one("#candidate-input", Input).value.strip()
            candidates = [c.strip() for c in candidates_raw.splitlines() if c.strip()]
            if anchor and candidates:
                self._run_similarity(anchor, candidates)
        elif btn_id == "recall-btn":
            event.stop()
            self._run_recall()
        elif btn_id == "beir-btn":
            event.stop()
            dataset = self.query_one("#beir-select", Select).value
            self._run_beir(str(dataset))

    @work(thread=True)
    def _run_similarity(self, anchor: str, candidates: list[str]) -> None:
        import yaml
        from mlx_tune.embeddings import FastEmbeddingModel
        from evaluation.embedding_evaluator import compute_similarity

        ws = Path("workspaces") / self.domain
        cfg_path = ws / "runtime_model_config.yaml"
        if not cfg_path.exists():
            self._post_result("Generate runtime configs first (run training panel).")
            return
        m_cfg = yaml.safe_load(cfg_path.read_text()).get("embedding", {})
        model, tokenizer = FastEmbeddingModel.from_pretrained(
            m_cfg.get("base_model", "mlx-community/all-MiniLM-L6-v2")
        )
        adapters_dir = ws / "adapters"
        if adapters_dir.exists():
            model = FastEmbeddingModel.get_peft_model(model, r=m_cfg.get("lora", {}).get("rank", 16))

        scores = compute_similarity(anchor, candidates, model, tokenizer)
        ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        lines = [f"{score:.4f}  {cand}" for cand, score in ranked]
        self._post_result("\n".join(lines))

    @work(thread=True)
    def _run_recall(self) -> None:
        import yaml
        from mlx_tune.embeddings import FastEmbeddingModel
        from evaluation.embedding_evaluator import recall_at_k

        ws = Path("workspaces") / self.domain
        cfg_path = ws / "runtime_model_config.yaml"
        if not cfg_path.exists():
            self._post_result("Generate runtime configs first.")
            return
        m_cfg = yaml.safe_load(cfg_path.read_text()).get("embedding", {})
        model, tokenizer = FastEmbeddingModel.from_pretrained(
            m_cfg.get("base_model", "mlx-community/all-MiniLM-L6-v2")
        )
        val_path = ws / "processed" / "embedding_val.json"
        metrics = recall_at_k(val_path, model, tokenizer, k=[1, 5, 10])
        lines = [f"{k}: {v:.4f}" for k, v in metrics.items()]
        self._post_result("\n".join(lines))

    @work(thread=True)
    def _run_beir(self, dataset_name: str) -> None:
        import yaml
        from mlx_tune.embeddings import FastEmbeddingModel
        from evaluation.embedding_evaluator import run_beir

        ws = Path("workspaces") / self.domain
        cfg_path = ws / "runtime_model_config.yaml"
        if not cfg_path.exists():
            self._post_result("Generate runtime configs first.")
            return
        m_cfg = yaml.safe_load(cfg_path.read_text()).get("embedding", {})
        model, tokenizer = FastEmbeddingModel.from_pretrained(
            m_cfg.get("base_model", "mlx-community/all-MiniLM-L6-v2")
        )
        result = run_beir(dataset_name, model, tokenizer)
        if "error" in result:
            self._post_result(
                f"BEIR not available: {result['error']}\n"
                "Install with: pip install beir"
            )
        else:
            lines = [f"BEIR {dataset_name} {k}: {v:.4f}" for k, v in result.items()]
            self._post_result("\n".join(lines))

    def _post_result(self, text: str) -> None:
        self.call_from_thread(self._update_result, text)

    def _update_result(self, text: str) -> None:
        self.query_one("#eval-results", Label).update(text)
        self.query_one(LogView).write_line(text)
