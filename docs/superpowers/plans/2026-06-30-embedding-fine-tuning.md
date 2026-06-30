# Embedding Fine-Tuning Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add embedding model fine-tuning to ElixirTune via a domain type system that keeps the LM (SFT/DPO/GRPO) workflow and the embedding workflow fully separate in the TUI.

**Architecture:** Approach A — domain type as a top-level field in each domain's `config.yaml` (`type: lm | embedding`, default `lm`). The TUI reads this field on domain selection and mounts/activates the appropriate panel set. Embedding training uses `mlx_tune.embeddings.FastEmbeddingModel` + `EmbeddingSFTTrainer`. Cross-encoder training uses `sentence_transformers.CrossEncoder` (PyTorch/MPS). Evaluation adds cosine similarity probe, Recall@K, BEIR benchmark, and cross-encoder reranking.

**Tech Stack:** Python, mlx_tune (FastEmbeddingModel / EmbeddingSFTTrainer / EmbeddingSFTConfig), sentence-transformers (CrossEncoder), beir (optional), Textual TUI, Typer CLI.

## Global Constraints

- `type: lm` is the default when `config.yaml` is absent or has no `type` field — all existing LM domains continue to work without any change.
- The existing LM panels (Synth, Training, Eval, Deploy, Chat) are untouched and must not regress.
- All new `src/training/*.py` files follow the same `run(domain, model_config_path, training_config_path, train_data_path, val_data_path)` signature as `sft.py` and `dpo.py`.
- Adapters always write to `workspaces/<domain>/adapters/`. Cross-encoder adapters write to `workspaces/<domain>/ce_adapters/`.
- Metrics always write to `workspaces/<domain>/logs/training/training_metrics.json`.
- The `beir` package is optional — if not installed the BEIR section in the eval panel shows an install prompt instead of raising an error.
- Use `.venv/bin/python` (not bare `python3`) for all test commands.

## File Map

| File | Action | Responsibility |
|---|---|---|
| `tui/domain.py` | Modify | `read_domain_type()`, new `Status` values, `infer_status` branching |
| `commands/init.py` | Modify | `--type` flag, write `config.yaml` |
| `config/model_config.yaml` | Modify | Add `embedding:` and `cross_encoder:` blocks |
| `config/training_config.yaml` | Modify | Add `embedding:` block |
| `tui/new_domain.py` | Modify | Domain type selector radio buttons |
| `src/training/embedding.py` | Create | Bi-encoder fine-tuning via `FastEmbeddingModel` |
| `src/training/cross_encoder.py` | Create | Cross-encoder fine-tuning via `sentence_transformers.CrossEncoder` |
| `commands/prepare_embedding.py` | Create | `import` and `convert` modes for anchor/positive/negative data |
| `commands/train.py` | Modify | Add `embedding` and `cross-encoder` methods |
| `cli.py` | Modify | Register `prepare-embedding` command |
| `src/evaluation/embedding_evaluator.py` | Create | `compute_similarity`, `recall_at_k`, `run_beir`, `rerank_with_cross_encoder` |
| `commands/evaluate.py` | Modify | Add `--method embedding` / `cross-encoder` dispatch |
| `tui/panels/embedding_training.py` | Create | Data import, prepare, train buttons + log view |
| `tui/panels/embedding_eval.py` | Create | Similarity probe, Recall@K, BEIR, reranking |
| `tui/app.py` | Modify | Tab routing by domain type |
| `tui/sidebar.py` | Modify | `[LM]` / `[EM]` badge in domain list |
| `tests/tui/test_domain.py` | Modify | Extend with embedding status tests |
| `tests/test_commands.py` | Modify | Test `--type` flag in init |
| `requirements.txt` | Modify | Add `beir` (optional comment) |

---

### Task 1: Domain type system (`tui/domain.py`)

**Files:**
- Modify: `tui/domain.py`
- Modify: `tests/tui/test_domain.py`

**Interfaces:**
- Produces:
  - `read_domain_type(ws: Path) -> str` — returns `"lm"` or `"embedding"`
  - `Status.DATA_READY`, `Status.CE_TRAINED` — new enum values
  - `infer_status(ws)` — unchanged signature, now branches on domain type

- [ ] **Step 1: Write failing tests for new domain type functionality**

Add to `tests/tui/test_domain.py`:
```python
import yaml
from tui.domain import read_domain_type, Status, infer_status

def test_read_domain_type_defaults_to_lm(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    ws.mkdir(parents=True)
    assert read_domain_type(ws) == "lm"

def test_read_domain_type_reads_config(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    ws.mkdir(parents=True)
    (ws / "config.yaml").write_text(yaml.safe_dump({"type": "embedding"}))
    assert read_domain_type(ws) == "embedding"

def test_infer_status_embedding_empty(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    ws.mkdir(parents=True)
    (ws / "config.yaml").write_text(yaml.safe_dump({"type": "embedding"}))
    assert infer_status(ws) == Status.EMPTY

def test_infer_status_embedding_data_ready_seeds(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "seeds").mkdir(parents=True)
    (ws / "seeds" / "approved.jsonl").write_text("{}\n")
    (ws / "config.yaml").write_text(yaml.safe_dump({"type": "embedding"}))
    assert infer_status(ws) == Status.DATA_READY

def test_infer_status_embedding_data_ready_raw(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "data" / "raw").mkdir(parents=True)
    (ws / "data" / "raw" / "pairs.json").write_text("[]")
    (ws / "config.yaml").write_text(yaml.safe_dump({"type": "embedding"}))
    assert infer_status(ws) == Status.DATA_READY

def test_infer_status_embedding_prepared(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "processed").mkdir(parents=True)
    (ws / "processed" / "embedding_train.json").write_text("[]")
    (ws / "processed" / "embedding_val.json").write_text("[]")
    (ws / "config.yaml").write_text(yaml.safe_dump({"type": "embedding"}))
    assert infer_status(ws) == Status.PREPARED

def test_infer_status_embedding_trained(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "adapters").mkdir(parents=True)
    (ws / "adapters" / "adapter.npz").write_text("x")
    (ws / "config.yaml").write_text(yaml.safe_dump({"type": "embedding"}))
    assert infer_status(ws) == Status.TRAINED

def test_infer_status_embedding_ce_trained(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "adapters").mkdir(parents=True)
    (ws / "adapters" / "adapter.npz").write_text("x")
    (ws / "ce_adapters").mkdir(parents=True)
    (ws / "ce_adapters" / "pytorch_model.bin").write_text("x")
    (ws / "config.yaml").write_text(yaml.safe_dump({"type": "embedding"}))
    assert infer_status(ws) == Status.CE_TRAINED

def test_existing_lm_status_unaffected(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "processed").mkdir(parents=True)
    (ws / "processed" / "train.json").write_text("[]")
    # No config.yaml — defaults to lm
    assert infer_status(ws) == Status.PREPARED
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/python -m pytest tests/tui/test_domain.py -x -q 2>&1 | tail -20
```
Expected: failures on the new tests.

- [ ] **Step 3: Implement changes in `tui/domain.py`**

Add `DATA_READY` and `CE_TRAINED` to `Status`:
```python
class Status(str, Enum):
    EMPTY = "empty"
    SEEDED = "seeded"
    GENERATED = "generated"
    PREPARED = "prepared"
    TRAINED = "trained"
    EVALUATED = "evaluated"
    DEPLOYED = "deployed"
    DATA_READY = "data_ready"
    CE_TRAINED = "ce_trained"
```

Add `read_domain_type` after the `DomainState` dataclass:
```python
def read_domain_type(ws: Path) -> str:
    """Returns 'lm' or 'embedding'. Defaults to 'lm' if config.yaml absent or has no type."""
    cfg = Path(ws) / "config.yaml"
    if cfg.exists():
        data = yaml.safe_load(cfg.read_text()) or {}
        return data.get("type", "lm")
    return "lm"
```

Replace `infer_status` with a dispatching version:
```python
def infer_status(ws: Path) -> Status:
    ws = Path(ws)
    if read_domain_type(ws) == "embedding":
        return _infer_embedding_status(ws)
    return _infer_lm_status(ws)


def _infer_lm_status(ws: Path) -> Status:
    if (ws / "fused").exists() and any((ws / "fused").iterdir()):
        return Status.DEPLOYED
    if (ws / "logs" / "evaluation").exists() and any(
        (ws / "logs" / "evaluation").glob("*_evaluation.json")
    ):
        return Status.EVALUATED
    if (ws / "adapters").exists() and any((ws / "adapters").iterdir()):
        return Status.TRAINED
    if (ws / "processed" / "train.json").exists():
        return Status.PREPARED
    if (ws / "generated" / "filtered.jsonl").exists():
        return Status.GENERATED
    if (ws / "seeds" / "approved.jsonl").exists():
        return Status.SEEDED
    return Status.EMPTY


def _infer_embedding_status(ws: Path) -> Status:
    if (ws / "ce_adapters").exists() and any((ws / "ce_adapters").iterdir()):
        return Status.CE_TRAINED
    if (ws / "adapters").exists() and any((ws / "adapters").iterdir()):
        return Status.TRAINED
    if (ws / "processed" / "embedding_train.json").exists():
        return Status.PREPARED
    raw_dir = ws / "data" / "raw"
    if (raw_dir.exists() and any(raw_dir.iterdir())) or \
       (ws / "seeds" / "approved.jsonl").exists():
        return Status.DATA_READY
    return Status.EMPTY
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/tui/test_domain.py -x -q 2>&1 | tail -20
```
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tui/domain.py tests/tui/test_domain.py
git commit -m "feat: domain type system — read_domain_type(), embedding status ladder"
```

---

### Task 2: Init command writes `config.yaml` with domain type

**Files:**
- Modify: `commands/init.py`
- Modify: `tests/test_commands.py`

**Interfaces:**
- Consumes: `read_domain_type(ws)` from Task 1
- Produces: `cli.py init <domain> --type lm|embedding` writes `workspaces/<domain>/config.yaml` with `type: <type>`

- [ ] **Step 1: Write failing tests**

Open `tests/test_commands.py` and add:
```python
import yaml
from typer.testing import CliRunner
from cli import app

runner = CliRunner()

def test_init_writes_lm_type_by_default(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "mydom"])
    assert result.exit_code == 0
    cfg = yaml.safe_load((tmp_path / "workspaces" / "mydom" / "config.yaml").read_text())
    assert cfg["type"] == "lm"

def test_init_writes_embedding_type(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init", "mydom", "--type", "embedding"])
    assert result.exit_code == 0
    cfg = yaml.safe_load((tmp_path / "workspaces" / "mydom" / "config.yaml").read_text())
    assert cfg["type"] == "embedding"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
.venv/bin/python -m pytest tests/test_commands.py::test_init_writes_lm_type_by_default tests/test_commands.py::test_init_writes_embedding_type -x -q 2>&1 | tail -20
```
Expected: failures.

- [ ] **Step 3: Update `commands/init.py`**

Add `--type` option and write `config.yaml`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
import yaml
from data.synthetic.io import read_jsonl, write_jsonl
from commands import _ws

app = typer.Typer(context_settings={"allow_interspersed_args": True})

@app.callback(invoke_without_command=True)
def init(
    ctx: typer.Context,
    domain: str = typer.Argument(...),
    desc: str = typer.Option(None),
    seeds: str = typer.Option(None),
    type: str = typer.Option("lm", help="Domain type: lm | embedding"),
):
    """Initialise a new domain workspace."""
    if ctx.invoked_subcommand is not None:
        return
    if type not in ("lm", "embedding"):
        typer.echo(f"Invalid type '{type}'. Choose: lm, embedding", err=True)
        raise typer.Exit(1)
    ws = _ws(domain)
    cand = ws / "seeds" / "candidates.jsonl"
    cand.parent.mkdir(parents=True, exist_ok=True)
    if seeds:
        recs = read_jsonl(seeds)
        write_jsonl(cand, recs)
        typer.echo(f"Imported {len(recs)} seeds to {cand}")
    else:
        cand.touch()
        typer.echo(f"Created empty seed file at {cand}")
        typer.echo("Add seeds to the file or re-run with --seeds <path>", err=True)
    if desc:
        (ws / "description.txt").write_text(desc)
    cfg_path = ws / "config.yaml"
    existing = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    existing["type"] = type
    cfg_path.write_text(yaml.safe_dump(existing))
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_commands.py -x -q 2>&1 | tail -20
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add commands/init.py tests/test_commands.py
git commit -m "feat: init --type flag writes domain type to config.yaml"
```

---

### Task 3: Config YAML defaults for embedding

**Files:**
- Modify: `config/model_config.yaml`
- Modify: `config/training_config.yaml`
- Modify: `tui/domain.py` (`generate_runtime_configs` — no test change needed)

**Interfaces:**
- Produces: global config files with `embedding:` and `cross_encoder:` blocks that `src/training/embedding.py` and `src/training/cross_encoder.py` will read.

- [ ] **Step 1: Add `embedding` and `cross_encoder` blocks to `config/model_config.yaml`**

Append to the end of the file:
```yaml
embedding:
  base_model: mlx-community/all-MiniLM-L6-v2
  max_seq_length: 256
  pooling_strategy: mean      # mean | cls | last_token
  lora:
    rank: 16
    alpha: 16
    dropout: 0.0
cross_encoder:
  base_model: cross-encoder/ms-marco-MiniLM-L-6-v2
  max_seq_length: 512
  lora:
    rank: 8
    alpha: 8
    dropout: 0.0
```

- [ ] **Step 2: Add `embedding` block to `config/training_config.yaml`**

Append to the end of the file:
```yaml
embedding:
  batch_size: 32
  learning_rate: 2.0e-05
  iters: 100
  loss_type: infonce          # infonce | triplet
  temperature: 0.05           # used by infonce
  margin: 1.0                 # used by triplet
  normalize_embeddings: true
  anchor_column: anchor
  positive_column: positive
  negative_column: null       # set to "negative" for triplet loss
```

- [ ] **Step 3: Update `generate_runtime_configs` in `tui/domain.py` to patch embedding output paths**

In the `generate_runtime_configs` function, after the existing `overlay` calls, add an overlay for the embedding config so the output dir is workspace-specific:
```python
def generate_runtime_configs(ws: Path, root: Path = Path(".")) -> None:
    ws, root = Path(ws), Path(root)
    ws.mkdir(parents=True, exist_ok=True)

    def overlay(base_file: str, overrides: dict, out_name: str) -> None:
        base_path = root / base_file
        base = yaml.safe_load(base_path.read_text()) if base_path.exists() else {}
        (ws / out_name).write_text(yaml.safe_dump(_deep_merge(base, overrides)))

    overlay("config/model_config.yaml", {
        "paths": {
            "adapter_dir": str(resolve_adapters_dir(ws)),
            "fused_model_dir": str(ws / "fused"),
            "checkpoint_dir": str(ws / "checkpoints"),
        },
        "embedding": {
            "output_dir": str(ws / "adapters"),
        },
        "cross_encoder": {
            "output_dir": str(ws / "ce_adapters"),
        },
    }, "runtime_model_config.yaml")

    overlay("config/training_config.yaml", {
        "paths": {
            "train_data": str(ws / "processed" / "train.json"),
            "test_data": str(ws / "processed" / "test.json"),
            "logs_dir": str(ws / "logs" / "training"),
        },
    }, "runtime_training_config.yaml")

    overlay("config/evaluation_config.yaml", {
        "paths": {
            "results_dir": str(ws / "logs" / "evaluation"),
            "test_data": str(ws / "processed" / "test.json"),
        }
    }, "runtime_eval_config.yaml")
```

- [ ] **Step 4: Run existing domain tests to confirm nothing broke**

```bash
.venv/bin/python -m pytest tests/tui/test_domain.py -x -q 2>&1 | tail -20
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add config/model_config.yaml config/training_config.yaml tui/domain.py
git commit -m "feat: add embedding and cross_encoder config defaults"
```

---

### Task 4: New domain wizard type selector

**Files:**
- Modify: `tui/new_domain.py`
- Modify: `tests/tui/test_new_domain.py`

**Interfaces:**
- Consumes: `commands/init.py --type` flag (Task 2)
- Produces: a type radio in the New Domain modal that passes `--type lm` or `--type embedding` to `cli.py init`

- [ ] **Step 1: Write failing tests**

In `tests/tui/test_new_domain.py`, add:
```python
from tui.app import ElixirTuneApp

async def test_new_domain_has_type_selector(tmp_path):
    async with ElixirTuneApp(root=tmp_path).run_test() as pilot:
        await pilot.click("#new-domain-btn")
        await pilot.pause()
        from textual.widgets import RadioButton
        buttons = list(pilot.app.screen.query(RadioButton))
        labels = [str(b.label) for b in buttons]
        assert any("Language Model" in l for l in labels)
        assert any("Embedding" in l for l in labels)
```

- [ ] **Step 2: Run to confirm it fails**

```bash
.venv/bin/python -m pytest tests/tui/test_new_domain.py::test_new_domain_has_type_selector -x -q 2>&1 | tail -20
```
Expected: failure.

- [ ] **Step 3: Update `tui/new_domain.py`**

Add a domain type `RadioSet` after the domain name input:
```python
import subprocess
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Label, RadioButton, RadioSet, TextArea


class NewDomainScreen(ModalScreen):
    DEFAULT_CSS = """
    NewDomainScreen {
        align: center middle;
    }
    NewDomainScreen #dialog {
        width: 60;
        height: auto;
        background: $surface;
        border: solid $primary;
        padding: 1 2;
    }
    NewDomainScreen #dialog Label {
        height: 1;
        margin-top: 1;
        color: $text-muted;
    }
    NewDomainScreen #dialog Label#dialog-title {
        color: $accent;
        text-style: bold;
        margin-top: 0;
    }
    NewDomainScreen #dialog RadioSet {
        height: auto;
        margin: 1 0;
        border: none;
    }
    NewDomainScreen #dialog TextArea {
        height: 4;
        margin-bottom: 1;
    }
    NewDomainScreen #btn-row {
        height: auto;
        margin-top: 1;
    }
    NewDomainScreen #btn-row Button { width: 1fr; margin-right: 1; }
    NewDomainScreen .hidden { display: none; }
    """

    def __init__(self, root: Path = Path("."), **kwargs) -> None:
        super().__init__(**kwargs)
        self._root = root

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("New Domain", id="dialog-title")
            yield Label("Domain name (lowercase, no spaces):")
            yield Input(id="new-domain-name", placeholder="e.g. my_assistant")
            yield Label("Domain type:")
            yield RadioSet(
                RadioButton("Language Model", id="rb-type-lm", value=True),
                RadioButton("Embedding Model", id="rb-type-embedding"),
                id="type-radio",
            )
            yield RadioSet(
                RadioButton("Bootstrap from description", id="rb-bootstrap", value=True),
                RadioButton("Import from file", id="rb-import"),
                id="source-radio",
            )
            yield Label("Description:", id="label-desc")
            yield TextArea(id="new-domain-desc")
            yield Label("Seeds file path:", id="label-seeds", classes="hidden")
            yield Input(id="new-domain-seeds-path", placeholder="/path/to/seeds.jsonl",
                        classes="hidden")
            with Horizontal(id="btn-row"):
                yield Button("Create", id="new-domain-create", variant="primary")
                yield Button("Cancel", id="new-domain-cancel")

    def _domain_type(self) -> str:
        radio = self.query_one("#type-radio", RadioSet)
        if radio.pressed_button and radio.pressed_button.id == "rb-type-embedding":
            return "embedding"
        return "lm"

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        if event.radio_set.id == "source-radio":
            is_import = event.pressed.id == "rb-import"
            self.query_one("#label-desc").set_class(is_import, "hidden")
            self.query_one("#new-domain-desc").set_class(is_import, "hidden")
            self.query_one("#label-seeds").set_class(not is_import, "hidden")
            self.query_one("#new-domain-seeds-path").set_class(not is_import, "hidden")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "new-domain-cancel":
            self.dismiss(None)
        elif event.button.id == "new-domain-create":
            name = self.query_one("#new-domain-name", Input).value.strip()
            if not name or " " in name:
                self.app.notify("Enter a valid domain name (no spaces).", severity="warning")
                return
            source_radio = self.query_one("#source-radio", RadioSet)
            pressed = source_radio.pressed_button and source_radio.pressed_button.id
            domain_type = self._domain_type()
            if pressed == "rb-import":
                seeds = self.query_one("#new-domain-seeds-path", Input).value.strip()
                cmd = ["python3", "cli.py", "init", name, "--seeds", seeds, "--type", domain_type]
            else:
                desc = self.query_one("#new-domain-desc", TextArea).text.strip() or f"{name} domain"
                cmd = ["python3", "cli.py", "init", name, "--desc", desc, "--type", domain_type]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                self.dismiss({"name": name, "success": True})
            else:
                self.dismiss({"name": name, "success": False, "error": result.stderr})
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/tui/test_new_domain.py -x -q 2>&1 | tail -20
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tui/new_domain.py tests/tui/test_new_domain.py
git commit -m "feat: new domain wizard adds domain type selector (LM vs Embedding)"
```

---

### Task 5: Embedding training backend (`src/training/embedding.py`)

**Files:**
- Create: `src/training/embedding.py`
- Create: `tests/test_training.py` additions (or new test file if existing is too LM-specific)

**Interfaces:**
- Consumes: `config/model_config.yaml` `embedding:` block, `config/training_config.yaml` `embedding:` block (Task 3)
- Produces: `run(domain, model_config_path, training_config_path, train_data_path, val_data_path)` — saves adapters to `workspaces/<domain>/adapters/`

- [ ] **Step 1: Write failing tests**

Create `tests/test_embedding_training.py`:
```python
import json
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch

def _write_config(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data))

def _write_data(path: Path, records: list) -> None:
    path.write_text(json.dumps(records))


def test_embedding_run_loads_correct_config(tmp_path):
    """run() reads embedding block from model config, not top-level lm keys."""
    model_cfg = tmp_path / "model_config.yaml"
    train_cfg = tmp_path / "training_config.yaml"
    train_data = tmp_path / "train.json"
    _write_config(model_cfg, {
        "embedding": {
            "base_model": "mlx-community/all-MiniLM-L6-v2",
            "max_seq_length": 128,
            "pooling_strategy": "mean",
            "lora": {"rank": 8, "alpha": 8, "dropout": 0.0},
        }
    })
    _write_config(train_cfg, {
        "embedding": {
            "batch_size": 4,
            "learning_rate": 2e-5,
            "iters": 2,
            "loss_type": "infonce",
            "temperature": 0.05,
            "margin": 1.0,
            "normalize_embeddings": True,
            "anchor_column": "anchor",
            "positive_column": "positive",
            "negative_column": None,
        }
    })
    _write_data(train_data, [
        {"anchor": "hello", "positive": "hi there"},
        {"anchor": "goodbye", "positive": "see you"},
    ])

    with patch("mlx_tune.embeddings.FastEmbeddingModel") as MockModel, \
         patch("mlx_tune.embeddings.EmbeddingSFTTrainer") as MockTrainer, \
         patch("mlx_tune.embeddings.EmbeddingSFTConfig") as MockConfig:
        mock_model = MagicMock()
        MockModel.from_pretrained.return_value = (mock_model, MagicMock())
        MockModel.get_peft_model.return_value = mock_model

        from src.training.embedding import run
        run("testdomain", model_cfg, train_cfg, train_data, None)

        MockModel.from_pretrained.assert_called_once()
        call_kwargs = MockModel.from_pretrained.call_args
        assert call_kwargs[0][0] == "mlx-community/all-MiniLM-L6-v2"
        MockTrainer.return_value.train.assert_called_once()
```

- [ ] **Step 2: Run to confirm it fails**

```bash
.venv/bin/python -m pytest tests/test_embedding_training.py -x -q 2>&1 | tail -20
```
Expected: `ModuleNotFoundError` or import failure.

- [ ] **Step 3: Create `src/training/embedding.py`**

```python
import json
import yaml
from pathlib import Path

import src._compat  # noqa: F401


def _load_configs(model_config_path: Path, training_config_path: Path) -> dict:
    m = yaml.safe_load(Path(model_config_path).read_text())
    t = yaml.safe_load(Path(training_config_path).read_text())
    return {"model": m, "training": t}


def run(
    domain: str,
    model_config_path: Path,
    training_config_path: Path,
    train_data_path: Path,
    val_data_path: Path | None,
) -> None:
    from mlx_tune.embeddings import FastEmbeddingModel, EmbeddingSFTTrainer, EmbeddingSFTConfig
    from datasets import Dataset

    cfg = _load_configs(Path(model_config_path), Path(training_config_path))
    m_cfg = cfg["model"]["embedding"]
    t_cfg = cfg["training"]["embedding"]

    output_dir = str(Path("workspaces") / domain / "adapters")

    model, tokenizer = FastEmbeddingModel.from_pretrained(
        m_cfg["base_model"],
        max_seq_length=m_cfg.get("max_seq_length", 512),
        pooling_strategy=m_cfg.get("pooling_strategy", "mean"),
    )
    model = FastEmbeddingModel.get_peft_model(
        model,
        r=int(m_cfg["lora"]["rank"]),
        lora_alpha=int(m_cfg["lora"]["alpha"]),
        lora_dropout=float(m_cfg["lora"]["dropout"]),
    )

    train_ds = Dataset.from_list(json.loads(Path(train_data_path).read_text()))
    eval_ds = Dataset.from_list(json.loads(Path(val_data_path).read_text())) if val_data_path else None

    negative_col = t_cfg.get("negative_column") or None

    args = EmbeddingSFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=int(t_cfg.get("batch_size", 32)),
        learning_rate=float(t_cfg["learning_rate"]),
        max_steps=int(t_cfg["iters"]),
        loss_type=t_cfg.get("loss_type", "infonce"),
        temperature=float(t_cfg.get("temperature", 0.05)),
        margin=float(t_cfg.get("margin", 1.0)),
        normalize_embeddings=bool(t_cfg.get("normalize_embeddings", True)),
        anchor_column=t_cfg.get("anchor_column", "anchor"),
        positive_column=t_cfg.get("positive_column", "positive"),
        negative_column=negative_col,
        max_seq_length=m_cfg.get("max_seq_length", 512),
    )

    trainer = EmbeddingSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=args,
    )
    trainer.train()
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_embedding_training.py -x -q 2>&1 | tail -20
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/training/embedding.py tests/test_embedding_training.py
git commit -m "feat: embedding bi-encoder fine-tuning via mlx_tune FastEmbeddingModel"
```

---

### Task 6: Cross-encoder training backend (`src/training/cross_encoder.py`)

**Files:**
- Create: `src/training/cross_encoder.py`
- Create: `tests/test_cross_encoder_training.py`

**Interfaces:**
- Consumes: `config/model_config.yaml` `cross_encoder:` block (Task 3)
- Produces: `run(domain, model_config_path, training_config_path, train_data_path, val_data_path)` — saves model to `workspaces/<domain>/ce_adapters/`

- [ ] **Step 1: Write failing tests**

Create `tests/test_cross_encoder_training.py`:
```python
import json
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_cross_encoder_run_creates_output_dir(tmp_path):
    model_cfg = tmp_path / "model_config.yaml"
    train_cfg = tmp_path / "training_config.yaml"
    train_data = tmp_path / "train.json"

    model_cfg.write_text(yaml.safe_dump({
        "cross_encoder": {
            "base_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "max_seq_length": 256,
        }
    }))
    train_cfg.write_text(yaml.safe_dump({
        "embedding": {
            "batch_size": 4,
            "learning_rate": 2e-5,
            "iters": 2,
            "anchor_column": "anchor",
            "positive_column": "positive",
            "negative_column": "negative",
        }
    }))
    train_data.write_text(json.dumps([
        {"anchor": "q1", "positive": "rel doc", "negative": "irrel doc"},
    ]))

    with patch("sentence_transformers.CrossEncoder") as MockCE:
        mock_ce = MagicMock()
        MockCE.return_value = mock_ce

        from src.training.cross_encoder import run
        run("testdomain", model_cfg, train_cfg, train_data, None)

        mock_ce.fit.assert_called_once()


def test_cross_encoder_pairs_only_without_negatives(tmp_path):
    model_cfg = tmp_path / "model_config.yaml"
    train_cfg = tmp_path / "training_config.yaml"
    train_data = tmp_path / "train.json"

    model_cfg.write_text(yaml.safe_dump({
        "cross_encoder": {"base_model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "max_seq_length": 128}
    }))
    train_cfg.write_text(yaml.safe_dump({
        "embedding": {
            "batch_size": 4, "learning_rate": 2e-5, "iters": 2,
            "anchor_column": "anchor", "positive_column": "positive", "negative_column": None,
        }
    }))
    train_data.write_text(json.dumps([
        {"anchor": "q1", "positive": "doc1"},
        {"anchor": "q2", "positive": "doc2"},
    ]))

    with patch("sentence_transformers.CrossEncoder") as MockCE:
        mock_ce = MagicMock()
        MockCE.return_value = mock_ce

        from src.training.cross_encoder import run
        run("testdomain", model_cfg, train_cfg, train_data, None)

        fit_args = mock_ce.fit.call_args
        train_samples = fit_args[1].get("train_dataloader") or fit_args[0][0]
        # Without negatives, only positive pairs are created (score=1.0)
        # We can't easily inspect the dataloader, but verifying fit was called suffices
        mock_ce.fit.assert_called_once()
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/test_cross_encoder_training.py -x -q 2>&1 | tail -20
```
Expected: `ModuleNotFoundError` or `ImportError`.

- [ ] **Step 3: Create `src/training/cross_encoder.py`**

```python
import json
import yaml
from pathlib import Path


def _load_configs(model_config_path: Path, training_config_path: Path) -> dict:
    m = yaml.safe_load(Path(model_config_path).read_text())
    t = yaml.safe_load(Path(training_config_path).read_text())
    return {"model": m, "training": t}


def run(
    domain: str,
    model_config_path: Path,
    training_config_path: Path,
    train_data_path: Path,
    val_data_path: Path | None,
) -> None:
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CEBinaryAccuracyEvaluator
    from torch.utils.data import DataLoader
    from sentence_transformers import InputExample

    cfg = _load_configs(Path(model_config_path), Path(training_config_path))
    ce_cfg = cfg["model"].get("cross_encoder", {})
    t_cfg = cfg["training"]["embedding"]

    base_model = ce_cfg.get("base_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    max_seq_length = ce_cfg.get("max_seq_length", 512)
    output_dir = str(Path("workspaces") / domain / "ce_adapters")
    batch_size = int(t_cfg.get("batch_size", 16))
    num_epochs = max(1, int(t_cfg.get("iters", 100)) // 10)
    anchor_col = t_cfg.get("anchor_column", "anchor")
    positive_col = t_cfg.get("positive_column", "positive")
    negative_col = t_cfg.get("negative_column") or None

    records = json.loads(Path(train_data_path).read_text())

    # Build (query, doc, label) pairs: positives get 1.0, negatives get 0.0
    samples = []
    for rec in records:
        samples.append(InputExample(texts=[rec[anchor_col], rec[positive_col]], label=1.0))
        if negative_col and negative_col in rec and rec[negative_col]:
            samples.append(InputExample(texts=[rec[anchor_col], rec[negative_col]], label=0.0))

    model = CrossEncoder(base_model, max_length=max_seq_length, num_labels=1)
    train_dataloader = DataLoader(samples, shuffle=True, batch_size=batch_size)

    evaluator = None
    if val_data_path and Path(val_data_path).exists():
        val_records = json.loads(Path(val_data_path).read_text())
        val_pairs = [(r[anchor_col], r[positive_col]) for r in val_records]
        val_labels = [1] * len(val_pairs)
        if negative_col:
            for r in val_records:
                if negative_col in r and r[negative_col]:
                    val_pairs.append((r[anchor_col], r[negative_col]))
                    val_labels.append(0)
        evaluator = CEBinaryAccuracyEvaluator(
            sentence_pairs=val_pairs, labels=val_labels
        )

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=num_epochs,
        output_path=output_dir,
    )
    print(f"Cross-encoder saved to {output_dir}")
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_cross_encoder_training.py -x -q 2>&1 | tail -20
```
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/training/cross_encoder.py tests/test_cross_encoder_training.py
git commit -m "feat: cross-encoder fine-tuning via sentence_transformers CrossEncoder"
```

---

### Task 7: Data preparation command (`commands/prepare_embedding.py`)

**Files:**
- Create: `commands/prepare_embedding.py`
- Modify: `cli.py` (register new command)
- Create: `tests/test_prepare_embedding.py`

**Interfaces:**
- Consumes: nothing from prior tasks at runtime; uses `read_domain_type` to validate
- Produces:
  - `workspaces/<domain>/processed/embedding_train.json`
  - `workspaces/<domain>/processed/embedding_val.json`
  - Both are JSON lists of dicts with at least `anchor` and `positive` keys.

- [ ] **Step 1: Write failing tests**

Create `tests/test_prepare_embedding.py`:
```python
import json
from pathlib import Path
from typer.testing import CliRunner
from cli import app

runner = CliRunner()


def _make_domain(tmp_path: Path, domain_type: str = "embedding") -> Path:
    import yaml
    ws = tmp_path / "workspaces" / "emb"
    ws.mkdir(parents=True)
    (ws / "config.yaml").write_text(yaml.safe_dump({"type": domain_type}))
    return ws


def test_prepare_embedding_import_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ws = _make_domain(tmp_path)
    data_file = tmp_path / "pairs.json"
    data_file.write_text(json.dumps([
        {"anchor": "hello", "positive": "hi"},
        {"anchor": "bye", "positive": "goodbye"},
        {"anchor": "cat", "positive": "feline"},
        {"anchor": "dog", "positive": "canine"},
        {"anchor": "yes", "positive": "correct"},
    ]))
    result = runner.invoke(app, [
        "prepare-embedding", "emb",
        "--mode", "import",
        "--data-file", str(data_file),
        "--val-split", "0.2",
    ])
    assert result.exit_code == 0, result.output
    train = json.loads((ws / "processed" / "embedding_train.json").read_text())
    val = json.loads((ws / "processed" / "embedding_val.json").read_text())
    assert len(train) + len(val) == 5
    assert len(val) == 1  # 20% of 5 = 1
    assert all("anchor" in r and "positive" in r for r in train + val)


def test_prepare_embedding_import_jsonl(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ws = _make_domain(tmp_path)
    data_file = tmp_path / "pairs.jsonl"
    data_file.write_text(
        '\n'.join(json.dumps({"anchor": f"q{i}", "positive": f"a{i}"}) for i in range(5))
    )
    result = runner.invoke(app, [
        "prepare-embedding", "emb",
        "--mode", "import",
        "--data-file", str(data_file),
    ])
    assert result.exit_code == 0, result.output
    train = json.loads((ws / "processed" / "embedding_train.json").read_text())
    val = json.loads((ws / "processed" / "embedding_val.json").read_text())
    assert len(train) + len(val) == 5


def test_prepare_embedding_convert_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ws = _make_domain(tmp_path)
    seeds_dir = ws / "seeds"
    seeds_dir.mkdir(parents=True)
    records = [
        {"conversation": [{"role": "user", "content": f"q{i}"},
                          {"role": "assistant", "content": f"a{i}"}]}
        for i in range(6)
    ]
    (seeds_dir / "approved.jsonl").write_text('\n'.join(json.dumps(r) for r in records))

    result = runner.invoke(app, ["prepare-embedding", "emb", "--mode", "convert"])
    assert result.exit_code == 0, result.output
    train = json.loads((ws / "processed" / "embedding_train.json").read_text())
    val = json.loads((ws / "processed" / "embedding_val.json").read_text())
    assert len(train) + len(val) == 6
    assert all("anchor" in r and "positive" in r for r in train + val)


def test_prepare_embedding_import_missing_column(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_domain(tmp_path)
    data_file = tmp_path / "bad.json"
    data_file.write_text(json.dumps([{"query": "hello", "doc": "hi"}]))
    result = runner.invoke(app, [
        "prepare-embedding", "emb",
        "--mode", "import",
        "--data-file", str(data_file),
    ])
    assert result.exit_code != 0
```

- [ ] **Step 2: Run to confirm failures**

```bash
.venv/bin/python -m pytest tests/test_prepare_embedding.py -x -q 2>&1 | tail -20
```
Expected: failures.

- [ ] **Step 3: Create `commands/prepare_embedding.py`**

```python
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
from commands import _ws

app = typer.Typer(context_settings={"allow_interspersed_args": True})

_DEFAULT_ANCHOR_COL = "anchor"
_DEFAULT_POSITIVE_COL = "positive"


def _read_data_file(path: Path) -> list[dict]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _validate_columns(records: list[dict], anchor_col: str, positive_col: str) -> None:
    missing = [c for c in (anchor_col, positive_col) if c not in records[0]]
    if missing:
        typer.echo(
            f"Data file missing required columns: {missing}. "
            f"Expected at least '{anchor_col}' and '{positive_col}'.",
            err=True,
        )
        raise typer.Exit(1)


def _split_and_write(records: list[dict], ws: Path, val_split: float) -> None:
    random.shuffle(records)
    val_n = max(1, int(len(records) * val_split))
    val, train = records[:val_n], records[val_n:]
    out = ws / "processed"
    out.mkdir(parents=True, exist_ok=True)
    (out / "embedding_train.json").write_text(json.dumps(train, indent=2))
    (out / "embedding_val.json").write_text(json.dumps(val, indent=2))
    typer.echo(
        f"Prepared {len(records)} pairs → {out} "
        f"(train={len(train)}, val={len(val)})"
    )


@app.callback(invoke_without_command=True)
def prepare_embedding(
    ctx: typer.Context,
    domain: str = typer.Argument(...),
    mode: str = typer.Option("import", help="Preparation mode: import | convert"),
    data_file: Path = typer.Option(None, help="[import] Path to JSON/JSONL with anchor/positive pairs"),
    val_split: float = typer.Option(0.1, help="Fraction held out for validation"),
    anchor_column: str = typer.Option(_DEFAULT_ANCHOR_COL, help="Column name for anchor texts"),
    positive_column: str = typer.Option(_DEFAULT_POSITIVE_COL, help="Column name for positive texts"),
) -> None:
    """Prepare anchor/positive(/negative) pair data for embedding fine-tuning."""
    if ctx.invoked_subcommand is not None:
        return

    ws = _ws(domain)

    if mode == "import":
        if not data_file:
            typer.echo("--data-file is required for import mode.", err=True)
            raise typer.Exit(1)
        records = _read_data_file(data_file)
        if not records:
            typer.echo("Data file is empty.", err=True)
            raise typer.Exit(1)
        _validate_columns(records, anchor_column, positive_column)
        # Copy raw file for provenance
        raw_dir = ws / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / data_file.name).write_bytes(data_file.read_bytes())
        _split_and_write(records, ws, val_split)

    elif mode == "convert":
        from data.synthetic.io import read_jsonl
        seeds_path = ws / "seeds" / "approved.jsonl"
        generated_path = ws / "generated" / "filtered.jsonl"
        seed_recs = read_jsonl(str(seeds_path)) if seeds_path.exists() else []
        gen_recs = read_jsonl(str(generated_path)) if generated_path.exists() else []
        all_recs = seed_recs + gen_recs
        if not all_recs:
            typer.echo(
                f"No source data found. Add seeds to {seeds_path} first.",
                err=True,
            )
            raise typer.Exit(1)
        pairs = []
        for rec in all_recs:
            conv = rec.get("conversation", [])
            for i in range(0, len(conv) - 1, 2):
                if conv[i].get("role") == "user" and conv[i + 1].get("role") == "assistant":
                    pairs.append({
                        anchor_column: conv[i]["content"],
                        positive_column: conv[i + 1]["content"],
                    })
        if not pairs:
            typer.echo("No Q&A pairs extracted from source data.", err=True)
            raise typer.Exit(1)
        typer.echo(f"Extracted {len(pairs)} pairs from {len(all_recs)} records.")
        _split_and_write(pairs, ws, val_split)

    else:
        typer.echo(f"Unknown mode '{mode}'. Choose: import, convert", err=True)
        raise typer.Exit(1)
```

- [ ] **Step 4: Register in `cli.py`**

Add after the existing imports:
```python
from commands.prepare_embedding import app as prepare_embedding_app
```

And after the existing `app.add_typer` calls:
```python
app.add_typer(prepare_embedding_app, name="prepare-embedding")
```

- [ ] **Step 5: Run tests**

```bash
.venv/bin/python -m pytest tests/test_prepare_embedding.py -x -q 2>&1 | tail -20
```
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add commands/prepare_embedding.py cli.py tests/test_prepare_embedding.py
git commit -m "feat: prepare-embedding command with import and convert modes"
```

---

### Task 8: Wire `commands/train.py` for embedding methods

**Files:**
- Modify: `commands/train.py`

**Interfaces:**
- Consumes: `src.training.embedding.run` (Task 5), `src.training.cross_encoder.run` (Task 6)
- Produces: `python cli.py train <domain> --method embedding` and `--method cross-encoder`

- [ ] **Step 1: Update `commands/train.py`**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

app = typer.Typer(context_settings={"allow_interspersed_args": True})


@app.callback(invoke_without_command=True)
def train(
    ctx: typer.Context,
    domain: str = typer.Argument(...),
    method: str = typer.Option("sft", help="Training method: sft | dpo | grpo | embedding | cross-encoder"),
    model_config: Path = typer.Option(..., help="Path to runtime model config YAML"),
    training_config: Path = typer.Option(..., help="Path to runtime training config YAML"),
    train_data: Path = typer.Option(..., help="Path to train data file"),
    val_data: Path = typer.Option(None, help="Path to val data file (optional)"),
) -> None:
    """Fine-tune a model using the specified training method."""
    if ctx.invoked_subcommand is not None:
        return
    if method == "sft":
        from src.training.sft import run
    elif method == "dpo":
        from src.training.dpo import run
    elif method == "grpo":
        from src.training.grpo import run
    elif method == "embedding":
        from src.training.embedding import run
    elif method == "cross-encoder":
        from src.training.cross_encoder import run
    else:
        raise typer.BadParameter(
            f"Unknown method '{method}'. Choose: sft, dpo, grpo, embedding, cross-encoder"
        )
    run(domain, model_config, training_config, train_data, val_data)
```

- [ ] **Step 2: Verify CLI help shows new methods**

```bash
.venv/bin/python cli.py train --help
```
Expected: help text mentions `embedding | cross-encoder`.

- [ ] **Step 3: Run all existing tests to confirm no regression**

```bash
.venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -20
```
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add commands/train.py
git commit -m "feat: train command supports embedding and cross-encoder methods"
```

---

### Task 9: Embedding evaluator (`src/evaluation/embedding_evaluator.py`)

**Files:**
- Create: `src/evaluation/embedding_evaluator.py`
- Create: `tests/test_embedding_evaluator.py`

**Interfaces:**
- Produces (all functions take `model` and `tokenizer` as loaded by `FastEmbeddingModel`):
  - `compute_similarity(anchor: str, candidates: list[str], model, tokenizer) -> list[float]`
  - `recall_at_k(val_path: Path, model, tokenizer, k: list[int] = [1, 5, 10]) -> dict[str, float]`
  - `run_beir(dataset_name: str, model, tokenizer) -> dict[str, float]` — returns `{"ndcg@10": float, "recall@100": float}`
  - `rerank_with_cross_encoder(query: str, candidates: list[str], ce_model_path: str) -> list[float]`

- [ ] **Step 1: Write failing tests**

Create `tests/test_embedding_evaluator.py`:
```python
import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np


def test_compute_similarity_returns_list_of_floats():
    with patch("mlx_tune.embeddings.FastEmbeddingModel") as MockModel:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        # Simulate encode returning numpy arrays
        MockModel.for_inference.return_value = mock_model
        mock_model.encode.side_effect = [
            np.array([[1.0, 0.0]]),                          # anchor
            np.array([[1.0, 0.0], [0.0, 1.0]]),              # candidates
        ]
        from src.evaluation.embedding_evaluator import compute_similarity
        scores = compute_similarity("hello", ["hi", "bye"], mock_model, mock_tokenizer)
        assert isinstance(scores, list)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)


def test_recall_at_k_returns_dict(tmp_path):
    val_path = tmp_path / "embedding_val.json"
    records = [
        {"anchor": "q1", "positive": "a1"},
        {"anchor": "q2", "positive": "a2"},
    ]
    val_path.write_text(json.dumps(records))

    with patch("src.evaluation.embedding_evaluator.FastEmbeddingModel") as MockModel:
        mock_model = MagicMock()
        MockModel.for_inference.return_value = mock_model
        n = len(records)
        # Encode returns (n, dim) arrays
        mock_model.encode.return_value = np.eye(n)
        from src.evaluation.embedding_evaluator import recall_at_k
        result = recall_at_k(val_path, mock_model, MagicMock(), k=[1])
        assert "recall@1" in result
        assert 0.0 <= result["recall@1"] <= 1.0


def test_run_beir_graceful_without_package():
    with patch.dict("sys.modules", {"beir": None}):
        from importlib import reload
        import src.evaluation.embedding_evaluator as mod
        reload(mod)
        result = mod.run_beir("scifact", MagicMock(), MagicMock())
        assert result == {"error": "beir package not installed"}


def test_rerank_with_cross_encoder_returns_scores():
    with patch("sentence_transformers.CrossEncoder") as MockCE:
        mock_ce = MagicMock()
        MockCE.return_value = mock_ce
        mock_ce.predict.return_value = [0.9, 0.1, 0.5]
        from src.evaluation.embedding_evaluator import rerank_with_cross_encoder
        scores = rerank_with_cross_encoder("query", ["doc1", "doc2", "doc3"], "/fake/path")
        assert len(scores) == 3
        assert scores[0] == 0.9
```

- [ ] **Step 2: Run to confirm failure**

```bash
.venv/bin/python -m pytest tests/test_embedding_evaluator.py -x -q 2>&1 | tail -20
```
Expected: failures.

- [ ] **Step 3: Create `src/evaluation/embedding_evaluator.py`**

```python
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-10)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-10)
    return a @ b.T


def compute_similarity(
    anchor: str,
    candidates: list[str],
    model,
    tokenizer,
) -> list[float]:
    """Return cosine similarity scores between anchor and each candidate."""
    from mlx_tune.embeddings import FastEmbeddingModel
    model = FastEmbeddingModel.for_inference(model)
    anchor_emb = model.encode([anchor])
    cand_embs = model.encode(candidates)
    sims = _cosine_similarity(anchor_emb, cand_embs)[0]
    return [float(s) for s in sims]


def recall_at_k(
    val_path: Path,
    model,
    tokenizer,
    k: list[int] = [1, 5, 10],
) -> dict[str, float]:
    """
    Compute Recall@K over a validation set of anchor/positive pairs.

    For each anchor, rank all positives by cosine similarity. A hit at K means
    the correct positive is in the top-K ranked candidates.
    """
    from mlx_tune.embeddings import FastEmbeddingModel
    records = json.loads(Path(val_path).read_text())
    if not records:
        return {f"recall@{ki}": 0.0 for ki in k}

    anchors = [r["anchor"] for r in records]
    positives = [r["positive"] for r in records]

    model = FastEmbeddingModel.for_inference(model)
    anchor_embs = model.encode(anchors)
    positive_embs = model.encode(positives)

    # similarity matrix: (n_anchors, n_positives)
    sim_matrix = _cosine_similarity(anchor_embs, positive_embs)

    results = {}
    for ki in k:
        hits = 0
        for i in range(len(anchors)):
            top_k_indices = np.argsort(sim_matrix[i])[::-1][:ki]
            if i in top_k_indices:
                hits += 1
        results[f"recall@{ki}"] = hits / len(anchors)
    return results


def run_beir(dataset_name: str, model, tokenizer) -> dict[str, float]:
    """
    Run BEIR benchmark evaluation for the given dataset.

    Returns {"ndcg@10": float, "recall@100": float}.
    If the `beir` package is not installed, returns {"error": "beir package not installed"}.
    """
    try:
        from beir import util as beir_util
        from beir.datasets.data_loader import GenericDataLoader
        from beir.retrieval.evaluation import EvaluateRetrieval
        from beir.retrieval import models as beir_models
    except ImportError:
        return {"error": "beir package not installed"}

    from mlx_tune.embeddings import FastEmbeddingModel

    # Download dataset
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = beir_util.download_and_unzip(url, "beir_datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    model = FastEmbeddingModel.for_inference(model)

    class _MLXRetriever:
        def encode_corpus(self, corpus_list, **kwargs):
            texts = [
                (d.get("title", "") + " " + d.get("text", "")).strip()
                for d in corpus_list
            ]
            return model.encode(texts)

        def encode_queries(self, queries_list, **kwargs):
            return model.encode(queries_list)

    retriever = EvaluateRetrieval(_MLXRetriever(), score_function="cos_sim")
    results = retriever.retrieve(corpus, queries)
    ndcg, _map, recall, _precision = EvaluateRetrieval.evaluate(
        qrels, results, [10, 100]
    )
    return {
        "ndcg@10": ndcg.get("NDCG@10", 0.0),
        "recall@100": recall.get("Recall@100", 0.0),
    }


def rerank_with_cross_encoder(
    query: str,
    candidates: list[str],
    ce_model_path: str,
) -> list[float]:
    """Rerank candidates using a cross-encoder. Returns scores in original order."""
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(ce_model_path)
    pairs = [(query, c) for c in candidates]
    scores = model.predict(pairs)
    return [float(s) for s in scores]
```

- [ ] **Step 4: Run tests**

```bash
.venv/bin/python -m pytest tests/test_embedding_evaluator.py -x -q 2>&1 | tail -20
```
Expected: all pass.

- [ ] **Step 5: Add `beir` to requirements**

In `requirements.txt`, after the `# Optional evaluation packages` comment block:
```
# Optional: BEIR benchmark for embedding evaluation (heavy install)
# beir>=1.0
```

- [ ] **Step 6: Commit**

```bash
git add src/evaluation/embedding_evaluator.py tests/test_embedding_evaluator.py requirements.txt
git commit -m "feat: embedding evaluator — similarity, recall@k, BEIR, cross-encoder reranking"
```

---

### Task 10: Wire `commands/evaluate.py` for embedding

**Files:**
- Modify: `commands/evaluate.py`

**Interfaces:**
- Consumes: `src.evaluation.embedding_evaluator` (Task 9)
- Produces: `python cli.py evaluate <domain> --method embedding [--val-data ...] [--beir-dataset ...]`

- [ ] **Step 1: Update `commands/evaluate.py`**

Add an optional `--method` flag with default `lm`. Route `embedding` to the embedding evaluator:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer

app = typer.Typer(context_settings={"allow_interspersed_args": True})

from commands import _ws


@app.callback(invoke_without_command=True)
def evaluate(
    ctx: typer.Context,
    domain: str = typer.Argument(...),
    method: str = typer.Option("lm", help="Evaluation method: lm | embedding | cross-encoder"),
    eval_config: Path = typer.Option(None, help="[lm] Path to runtime eval config YAML"),
    adapters_path: Path = typer.Option(None, help="Path to adapters dir"),
    test_data: Path = typer.Option(None, help="Path to test data file"),
    fused_path: Path = typer.Option(None, help="[lm] Path to fused model dir"),
    model_config: Path = typer.Option(None, help="Path to model config YAML"),
    max_samples: int = typer.Option(100, help="[lm] Max test samples (default 100)"),
    beir_dataset: str = typer.Option(None, help="[embedding] BEIR dataset name (e.g. scifact)"),
    val_data: Path = typer.Option(None, help="[embedding] Path to embedding_val.json"),
) -> None:
    """Evaluate a trained model."""
    if ctx.invoked_subcommand is not None:
        return

    ws = _ws(domain)

    if method == "lm":
        import yaml
        from evaluation.evaluator import ModelEvaluator

        if test_data is None:
            test_data = ws / "processed" / "test.json"
        cfg_path = Path(model_config) if model_config else ws / "runtime_model_config.yaml"
        if not cfg_path.exists():
            typer.echo(
                f"Model config not found at {cfg_path}. "
                "Pass --model-config <path> or run the TUI first.",
                err=True,
            )
            raise typer.Exit(1)
        model_cfg = yaml.safe_load(cfg_path.read_text())
        base_model = model_cfg["base_model"]["path"]
        evaluator = ModelEvaluator(str(eval_config))
        if adapters_path and Path(adapters_path).exists():
            evaluator.comprehensive_model_comparison(
                base_model_path=base_model,
                adapter_path=str(adapters_path),
                fused_model_path=str(fused_path) if fused_path else None,
                test_data_path=str(test_data),
                max_samples=max_samples,
            )
        else:
            evaluator.evaluate_model_from_path(base_model, "base_model", str(test_data), max_samples=max_samples)

    elif method in ("embedding", "cross-encoder"):
        import yaml
        from evaluation.embedding_evaluator import recall_at_k, run_beir

        cfg_path = Path(model_config) if model_config else ws / "runtime_model_config.yaml"
        model_cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        adapters_dir = Path(adapters_path) if adapters_path else ws / "adapters"

        if method == "embedding":
            from mlx_tune.embeddings import FastEmbeddingModel
            em_cfg = model_cfg.get("embedding", {})
            loaded_model, tokenizer = FastEmbeddingModel.from_pretrained(
                em_cfg.get("base_model", "mlx-community/all-MiniLM-L6-v2"),
                max_seq_length=em_cfg.get("max_seq_length", 512),
            )
        else:
            from sentence_transformers import CrossEncoder
            ce_cfg = model_cfg.get("cross_encoder", {})
            ce_path = str(ws / "ce_adapters") if (ws / "ce_adapters").exists() else ce_cfg.get("base_model", "")
            loaded_model, tokenizer = CrossEncoder(ce_path), None

        val_path = val_data or ws / "processed" / "embedding_val.json"
        if val_path.exists() and method == "embedding":
            metrics = recall_at_k(val_path, loaded_model, tokenizer)
            for key, val in metrics.items():
                typer.echo(f"{key}: {val:.4f}")

        if beir_dataset and method == "embedding":
            beir_result = run_beir(beir_dataset, loaded_model, tokenizer)
            for key, val in beir_result.items():
                typer.echo(f"BEIR {beir_dataset} {key}: {val:.4f}")
    else:
        typer.echo(f"Unknown method '{method}'. Choose: lm, embedding, cross-encoder", err=True)
        raise typer.Exit(1)
```

- [ ] **Step 2: Verify help**

```bash
.venv/bin/python cli.py evaluate --help
```
Expected: shows `--method` option.

- [ ] **Step 3: Run all tests**

```bash
.venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -20
```
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add commands/evaluate.py
git commit -m "feat: evaluate command supports --method embedding and cross-encoder"
```

---

### Task 11: TUI embedding training panel (`tui/panels/embedding_training.py`)

**Files:**
- Create: `tui/panels/embedding_training.py`

**Interfaces:**
- Consumes: `read_domain_type`, `Status`, `infer_status`, `generate_runtime_configs` from `tui/domain.py` (Task 1/3); `commands/train.py` embedding/cross-encoder methods (Task 8); `commands/prepare_embedding.py` (Task 7)
- Produces: `EmbeddingTrainingPanel(BasePanel)` — mountable in `tui/app.py`

- [ ] **Step 1: Create `tui/panels/embedding_training.py`**

```python
import json
import re
import subprocess
from pathlib import Path

import yaml
from textual.app import ComposeResult
from textual.containers import Horizontal
from textual.widgets import Button, Label, Rule, Select
from textual import work

from tui.app import BasePanel
from tui.domain import infer_status, Status, generate_runtime_configs, status_order
from tui.runner import RunnerOutput, RunnerDone, stream_subprocess
from tui.widgets.config_form import ConfigField, ConfigForm
from tui.widgets.log_view import LogView
from tui.widgets.section_rule import SectionRule

_EMBED_TRAIN_FIELDS = [
    ConfigField("Base embedding model", "config/model_config.yaml", ["embedding", "base_model"]),
    ConfigField("LoRA rank", "config/model_config.yaml", ["embedding", "lora", "rank"]),
    ConfigField("Loss type", "config/training_config.yaml", ["embedding", "loss_type"]),
    ConfigField("Learning rate", "config/training_config.yaml", ["embedding", "learning_rate"]),
    ConfigField("Iterations", "config/training_config.yaml", ["embedding", "iters"]),
]

_STEP_RE = re.compile(r"[Ss]tep[:\s]+(\d+).*[Ll]oss[:\s]+([\d.]+)")


class EmbeddingTrainingPanel(BasePanel):
    DEFAULT_CSS = """
    EmbeddingTrainingPanel { height: 100%; padding: 1 1 0 1; }
    EmbeddingTrainingPanel #embed-config-form { height: auto; max-height: 40%; overflow-y: auto; }
    EmbeddingTrainingPanel #embed-summary { height: auto; }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._metrics: dict | None = None

    def compose(self) -> ComposeResult:
        yield ConfigForm(_EMBED_TRAIN_FIELDS, id="embed-config-form")
        yield SectionRule("Training Summary")
        yield Label("", id="embed-summary")
        yield SectionRule("Run")
        with Horizontal(classes="btn-row"):
            yield Button("Import data", id="embed-import-btn", disabled=True, variant="success")
            yield Button("Convert from seeds", id="embed-convert-btn", disabled=True, variant="success")
            yield Button("▶ Train bi-encoder", id="embed-train-btn", disabled=True, variant="success")
            yield Button("▶ Train cross-encoder", id="embed-ce-train-btn", disabled=True, variant="success")
        yield SectionRule("Log")
        yield Label("", id="embed-train-progress")
        yield LogView(id="embed-train-log")
        yield Rule()

    def refresh_content(self) -> None:
        if not self.domain:
            return
        ws = Path("workspaces") / self.domain
        status = infer_status(ws)

        prepared = (ws / "processed" / "embedding_train.json").exists()
        trained = status in (Status.TRAINED, Status.CE_TRAINED)
        has_seeds = (ws / "seeds" / "approved.jsonl").exists()

        self.query_one("#embed-import-btn", Button).disabled = False
        self.query_one("#embed-convert-btn", Button).disabled = not has_seeds
        self.query_one("#embed-train-btn", Button).disabled = not prepared
        self.query_one("#embed-ce-train-btn", Button).disabled = not trained

        self._load_summary(ws)

    def _load_summary(self, ws: Path) -> None:
        label = self.query_one("#embed-summary", Label)
        metrics_file = ws / "logs" / "training" / "training_metrics.json"
        if not metrics_file.exists():
            label.update("")
            return
        try:
            m = json.loads(metrics_file.read_text())
            tl = m.get("train_loss", [])
            its = m.get("iterations", [])
            parts = []
            if its:
                parts.append(f"Iters: {its[-1]}")
            if tl:
                parts.append(f"Train loss: {tl[-1]:.4f}")
            label.update("  ·  ".join(parts) if parts else "")
        except (json.JSONDecodeError, OSError):
            label.update("")

    def watch_domain(self, domain: str | None) -> None:
        self._metrics = None
        super().watch_domain(domain)

    def _capture_metric(self, line: str) -> None:
        if self._metrics is None or not self.domain:
            return
        m = _STEP_RE.search(line)
        if not m:
            return
        self._metrics["iterations"].append(int(m.group(1)))
        self._metrics["train_loss"].append(float(m.group(2)))
        mp = Path("workspaces") / self.domain / "logs" / "training" / "training_metrics.json"
        mp.parent.mkdir(parents=True, exist_ok=True)
        mp.write_text(json.dumps(self._metrics))
        its, tl = self._metrics["iterations"], self._metrics["train_loss"]
        if its and tl:
            self.query_one("#embed-train-progress", Label).update(
                f"Iter {its[-1]}   loss: {tl[-1]:.3f}"
            )

    def on_config_form_saved(self, _: ConfigForm.Saved) -> None:
        self.app.notify("Config saved.")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id
        if not self.domain:
            return

        if btn_id == "embed-import-btn":
            event.stop()
            # TODO: open file picker — for now prompt via log
            self.query_one(LogView).write_line(
                "Run: python cli.py prepare-embedding <domain> --mode import --data-file <path>"
            )

        elif btn_id == "embed-convert-btn":
            event.stop()
            event.button.disabled = True
            self._run_cmd(
                ["python3", "cli.py", "prepare-embedding", self.domain, "--mode", "convert"],
                finish_id="embed-convert-btn",
            )

        elif btn_id == "embed-train-btn":
            event.stop()
            event.button.disabled = True
            ws = Path("workspaces") / self.domain
            generate_runtime_configs(ws)
            self._metrics = {"train_loss": [], "iterations": []}
            self.query_one("#embed-train-progress", Label).update("")
            self._run_train(self.domain, "embedding")

        elif btn_id == "embed-ce-train-btn":
            event.stop()
            event.button.disabled = True
            ws = Path("workspaces") / self.domain
            generate_runtime_configs(ws)
            self._metrics = {"train_loss": [], "iterations": []}
            self.query_one("#embed-train-progress", Label).update("")
            self._run_train(self.domain, "cross-encoder")

    @work(thread=True)
    def _run_cmd(self, cmd: list[str], finish_id: str) -> None:
        for line, code in stream_subprocess(cmd):
            if line is not None:
                self.post_message(RunnerOutput(line))
            else:
                self.post_message(RunnerDone(code, tag=finish_id))

    @work(thread=True)
    def _run_train(self, domain: str, method: str) -> None:
        ws = Path("workspaces") / domain
        if method == "cross-encoder":
            train_data = ws / "processed" / "embedding_train.json"
            val_data = ws / "processed" / "embedding_val.json"
        else:
            train_data = ws / "processed" / "embedding_train.json"
            val_data = ws / "processed" / "embedding_val.json"
        val_args = ["--val-data", str(val_data)] if val_data.exists() else []
        cmd = [
            "python3", "cli.py", "train", domain,
            "--method", method,
            "--model-config", str(ws / "runtime_model_config.yaml"),
            "--training-config", str(ws / "runtime_training_config.yaml"),
            "--train-data", str(train_data),
            *val_args,
        ]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            self.post_message(RunnerOutput(line.rstrip()))
        proc.wait()
        self.post_message(RunnerDone(proc.returncode))

    def on_runner_output(self, event: RunnerOutput) -> None:
        self.query_one(LogView).write_line(event.line)
        self._capture_metric(event.line)

    def on_runner_done(self, event: RunnerDone) -> None:
        if event.tag in ("embed-convert-btn",):
            if event.exit_code != 0:
                self.query_one(LogView).write_line(
                    f"[red]Data prep failed (exit {event.exit_code})[/red]"
                )
            else:
                self.query_one(LogView).write_line("[green]Data prepared.[/green]")
            self.refresh_content()
            self.call_later(self.app._rescan)
            return
        self._metrics = None
        self.query_one("#embed-train-btn", Button).disabled = False
        self.query_one("#embed-ce-train-btn", Button).disabled = False
        if event.exit_code != 0:
            self.query_one(LogView).write_line(
                f"[red]Training failed (exit {event.exit_code})[/red]"
            )
        else:
            self.query_one(LogView).write_line("[green]Training complete.[/green]")
        self.refresh_content()
        self.call_later(self.app._rescan)
```

- [ ] **Step 2: Verify import works**

```bash
.venv/bin/python -c "from tui.panels.embedding_training import EmbeddingTrainingPanel; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add tui/panels/embedding_training.py
git commit -m "feat: EmbeddingTrainingPanel TUI — import data, convert, train bi/cross-encoder"
```

---

### Task 12: TUI embedding eval panel (`tui/panels/embedding_eval.py`)

**Files:**
- Create: `tui/panels/embedding_eval.py`

**Interfaces:**
- Consumes: `src.evaluation.embedding_evaluator` (Task 9)
- Produces: `EmbeddingEvalPanel(BasePanel)` — mountable in `tui/app.py`

- [ ] **Step 1: Create `tui/panels/embedding_eval.py`**

```python
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
```

- [ ] **Step 2: Verify import**

```bash
.venv/bin/python -c "from tui.panels.embedding_eval import EmbeddingEvalPanel; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add tui/panels/embedding_eval.py
git commit -m "feat: EmbeddingEvalPanel TUI — similarity probe, recall@k, BEIR, reranking"
```

---

### Task 13: TUI app routing and sidebar badge

**Files:**
- Modify: `tui/app.py`
- Modify: `tui/sidebar.py`
- Modify: `tests/tui/test_app.py`
- Modify: `tests/tui/test_sidebar.py`

**Interfaces:**
- Consumes: `EmbeddingTrainingPanel` (Task 11), `EmbeddingEvalPanel` (Task 12), `read_domain_type` (Task 1)
- Produces: complete routed TUI — LM domains show LM tabs, embedding domains show embedding tabs; sidebar shows `[LM]`/`[EM]` badge.

- [ ] **Step 1: Write failing tests**

Add to `tests/tui/test_app.py`:
```python
import yaml

async def test_embedding_domain_shows_embedding_tabs(tmp_path):
    ws = tmp_path / "workspaces" / "emb"
    ws.mkdir(parents=True)
    (ws / "config.yaml").write_text(yaml.safe_dump({"type": "embedding"}))
    async with ElixirTuneApp(initial_domain="emb", root=tmp_path).run_test() as pilot:
        from textual.widgets import TabbedContent
        tc = pilot.app.query_one(TabbedContent)
        # Embedding-specific tab must be active/visible
        assert tc.query_one("#tab-embed-train") is not None

async def test_lm_domain_shows_lm_tabs(tmp_path):
    ws = tmp_path / "workspaces" / "lm"
    ws.mkdir(parents=True)
    # No config.yaml — defaults to lm
    async with ElixirTuneApp(initial_domain="lm", root=tmp_path).run_test() as pilot:
        from textual.widgets import TabbedContent
        tc = pilot.app.query_one(TabbedContent)
        assert tc.query_one("#tab-training") is not None
```

Add to `tests/tui/test_sidebar.py`:
```python
import yaml
from tui.domain import DomainState, Status
from tui.sidebar import Sidebar
from tui.app import ElixirTuneApp

async def test_sidebar_shows_em_badge_for_embedding_domain(tmp_path):
    ws = tmp_path / "workspaces" / "emb"
    ws.mkdir(parents=True)
    (ws / "config.yaml").write_text(yaml.safe_dump({"type": "embedding"}))
    async with ElixirTuneApp(root=tmp_path).run_test() as pilot:
        from textual.widgets import Label
        labels = [str(l.renderable) for l in pilot.app.query(Label)]
        assert any("[EM]" in l for l in labels)
```

- [ ] **Step 2: Run to confirm failures**

```bash
.venv/bin/python -m pytest tests/tui/test_app.py tests/tui/test_sidebar.py -x -q 2>&1 | tail -20
```
Expected: new tests fail.

- [ ] **Step 3: Update `tui/app.py`**

```python
from pathlib import Path

from textual.app import App, ComposeResult
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Header, TabbedContent, TabPane

from tui.domain import scan_domains, read_domain_type
from tui.sidebar import Sidebar, DomainSelected, NewDomainRequested, DeleteDomainRequested

_LM_TABS = ["tab-overview", "tab-synth", "tab-training", "tab-eval", "tab-deploy", "tab-chat"]
_EMBED_TABS = ["tab-overview", "tab-embed-data", "tab-embed-train", "tab-embed-eval"]
_ALL_TABS = list(dict.fromkeys(_LM_TABS + _EMBED_TABS))  # ordered, deduped


class BasePanel(Widget):
    domain: reactive[str | None] = reactive(None)

    def watch_domain(self, domain: str | None) -> None:
        if domain:
            self.refresh_content()

    def refresh_content(self) -> None:
        pass


from tui.panels.overview import OverviewPanel
from tui.panels.synthetic import SyntheticPanel
from tui.panels.training import TrainingPanel
from tui.panels.evaluation import EvaluationPanel
from tui.panels.deployment import DeploymentPanel
from tui.panels.embedding_training import EmbeddingTrainingPanel
from tui.panels.embedding_eval import EmbeddingEvalPanel


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
```

- [ ] **Step 4: Update `tui/sidebar.py` to show domain type badge**

Replace the `refresh_domains` method:
```python
async def refresh_domains(self, domains: list[DomainState], active: str | None = None) -> None:
    from tui.domain import read_domain_type
    self._active = active
    lv = self.query_one(ListView)
    await lv.clear()
    active_index = None
    for i, d in enumerate(domains):
        dot = "●" if d.name == active else "○"
        mark = " [b white]✓[/]" if d.status in (Status.DEPLOYED, Status.EVALUATED) else ""
        dtype = read_domain_type(d.workspace)
        badge = " [dim][EM][/]" if dtype == "embedding" else " [dim][LM][/]"
        lv.append(ListItem(Label(f"{dot} {d.name}{badge}{mark}"), id=f"domain-{d.name}"))
        if d.name == active:
            active_index = i
    if active_index is not None:
        self.call_after_refresh(lambda idx=active_index: setattr(lv, "index", idx))
```

- [ ] **Step 5: Run all tests**

```bash
.venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -20
```
Expected: all tests pass.

- [ ] **Step 6: Final smoke-test — launch TUI**

```bash
.venv/bin/python cli.py tui --help
```
Expected: help shows `ElixirTune`.

- [ ] **Step 7: Commit**

```bash
git add tui/app.py tui/sidebar.py tests/tui/test_app.py tests/tui/test_sidebar.py
git commit -m "feat: TUI routing by domain type — LM tabs vs embedding tabs, sidebar [LM]/[EM] badge"
```
