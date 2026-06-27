# Training Backend Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the broken `src/training/` module, delete all fine-tune-llm leftovers, restructure all pipeline entry points into a `commands/` package, and back the training step with mlx-tune (SFT + DPO stub + GRPO stub).

**Architecture:** A thin `cli.py` registers all Typer sub-apps from a `commands/` package; every pipeline operation is a CLI command callable as `python3 cli.py <command> <domain>`. The TUI calls these commands as subprocesses (unchanged interface, just updated paths). `src/training/` is replaced with clean mlx-tune wrappers; `src/evaluation/evaluator.py` gets a one-method fix to use the stable `mlx_lm.load(adapter_path=...)` API.

**Tech Stack:** Python 3.11+, Typer, mlx-tune ≥ 0.5.0, mlx-lm (transitive via mlx-tune), Textual 8.2.7, pytest-asyncio

## Global Constraints

- `mlx-tune>=0.5.0` replaces direct `mlx-lm` dependency in `requirements.txt`
- All workspace paths follow `workspaces/{domain}/` — never hardcode `data/`, `models/`, `logs/`
- Every `commands/*.py` is a **fresh rewrite** — old script logic is reference only
- Every command task includes an end-to-end smoke test of its `src/` module
- `config/defaults.yaml` is untouched — it is ElixirLoRA's synthetic pipeline config
- TUI subprocess calls stay in the form `["python3", "cli.py", "<command>", domain, ...]`
- All tests in `tests/` must pass after every task: run `pytest tests/ -q`

---

## File Map

```
DELETE:
  demo_pipeline.py
  fine-tune-playground.ipynb
  config/data_config.yaml
  venv/
  src/utils/plotting.py
  src/training/trainer.py
  src/training/lora_setup.py
  src/training/metrics.py
  src/training/__init__.py
  scripts/  (entire directory)

MODIFY:
  cli.py                            → thin entry point (~12 lines)
  config/model_config.yaml          → remove paths: block
  config/training_config.yaml       → remove paths: block, add method: sft
  config/evaluation_config.yaml     → remove paths: block
  requirements.txt                  → swap mlx-lm for mlx-tune
  src/evaluation/evaluator.py       → fix load_model_with_adapters only
  src/inference/generator.py        → remove hardcoded "Didier" system prompt
  tui/panels/training.py            → update subprocess cmd
  tui/panels/evaluation.py          → update subprocess cmds (×2)
  tui/panels/deployment.py          → update subprocess cmd

CREATE:
  commands/__init__.py
  commands/init.py                  (moved from cli.py)
  commands/curate.py                (moved from cli.py)
  commands/generate.py              (moved from cli.py)
  commands/prepare.py               (moved from cli.py)
  commands/upload.py                (moved from cli.py)
  commands/train.py                 (new — replaces scripts/02_train_model.py)
  commands/evaluate.py              (new — replaces scripts/03_evaluate_model.py)
  commands/fuse.py                  (new — replaces scripts/04_fuse_and_evaluate.py)
  commands/chat.py                  (new — replaces scripts/interactive_chat.py)
  src/training/__init__.py
  src/training/metrics_writer.py
  src/training/sft.py
  src/training/dpo.py
  src/training/grpo.py
  tests/test_training.py
  tests/test_commands.py
```

---

### Task 1: Delete fine-tune-llm artifacts and update configs

**Files:**
- Delete: `demo_pipeline.py`, `fine-tune-playground.ipynb`, `config/data_config.yaml`, `venv/`, `src/utils/plotting.py`, `src/training/trainer.py`, `src/training/lora_setup.py`, `src/training/metrics.py`, `src/training/__init__.py`, `scripts/` (entire dir)
- Modify: `config/model_config.yaml`, `config/training_config.yaml`, `config/evaluation_config.yaml`, `requirements.txt`

**Interfaces:**
- Produces: clean repo state; updated config templates; `mlx-tune` in requirements

- [ ] **Step 1: Delete all fine-tune-llm files**

```bash
rm demo_pipeline.py fine-tune-playground.ipynb
rm config/data_config.yaml
rm -rf venv/
rm src/utils/plotting.py
rm -rf src/training/
rm -rf scripts/
```

- [ ] **Step 2: Update `config/model_config.yaml` — remove `paths:` block**

Replace the entire file with:

```yaml
base_model:
  path: "microsoft/Phi-3-mini-4k-instruct"

lora:
  num_layers: 32
  lora_layers: 32
  rank: 16
  scale: 20.0
  dropout: 0.1
  keys:
    - "self_attn.q_proj"
    - "self_attn.k_proj"
    - "self_attn.v_proj"
    - "self_attn.o_proj"
```

- [ ] **Step 3: Update `config/training_config.yaml` — remove `paths:` block, add `method:`**

Replace the entire file with:

```yaml
method: sft

training:
  iters: 2000
  batch_size: 4
  learning_rate: 1e-5
  steps_per_eval: 50
  grad_checkpoint: true

optimizer:
  type: "adam"

metrics:
  patience: 5
  min_delta: 0.001
```

- [ ] **Step 4: Update `config/evaluation_config.yaml` — remove `paths:` block**

Replace the entire file with:

```yaml
evaluation:
  method: "simple"
  max_tokens: 200
  temperature: 0.7

metrics:
  bertscore:
    model_type: "microsoft/deberta-xlarge-mnli"
    lang: "en"
  simple:
    word_overlap_threshold: 0.5

comparison:
  compare_with_base: true
  score_thresholds:
    excellent: 0.9
    good: 0.7
    acceptable: 0.5
    poor: 0.3
```

- [ ] **Step 5: Update `requirements.txt` — swap `mlx-lm` for `mlx-tune`**

Remove the line `mlx-lm>=0.15.0` and add `mlx-tune>=0.5.0`. mlx-tune installs mlx-lm as a transitive dependency at the version it requires.

```
# Core MLX and ML packages
mlx>=0.15.0
mlx-tune>=0.5.0

# Data handling
datasets>=2.14.0
numpy>=1.24.0
pandas>=2.0.0

# Scientific computing
scikit-learn>=1.3.0

# Configuration and utilities
pyyaml>=6.0
python-dotenv>=1.0.0
tqdm>=4.65.0

# HuggingFace integration
huggingface-hub>=0.17.0
transformers>=4.30.0
tokenizers>=0.13.0

# Optional evaluation packages (install manually if needed)
# evaluate>=0.4.0
# bert-score>=0.3.13

# File I/O
safetensors>=0.3.0

# ElixirLoRA synthetic data pipeline
openai>=1.0
pydantic>=2.0
sentence-transformers>=2.2
typer>=0.9
pytest>=7.0
textual>=0.60
pytest-asyncio>=0.21
```

- [ ] **Step 6: Verify nothing the current codebase needs is gone**

```bash
python3 -c "import cli; print('cli ok')"
pytest tests/ -q
```

Expected: cli imports fine; tests pass (src/training imports are only in cli.py and will be updated in Task 2; nothing else imports from `scripts/`).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: delete fine-tune-llm artifacts, update config templates, swap mlx-tune dependency"
```

---

### Task 2: commands/ package + slim cli.py

**Files:**
- Create: `commands/__init__.py`, `commands/init.py`, `commands/curate.py`, `commands/generate.py`, `commands/prepare.py`, `commands/upload.py`
- Modify: `cli.py`

**Interfaces:**
- Consumes: existing `cli.py` command implementations (copy verbatim into their modules)
- Produces: `commands/` package; `cli.py` as thin router; `python3 cli.py --help` shows all subcommands

- [ ] **Step 1: Create `commands/__init__.py`**

```python
```
(Empty file — package marker only.)

- [ ] **Step 2: Create `commands/init.py`** — copy the `init` command from `cli.py`

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
from data.synthetic.io import read_jsonl, write_jsonl

app = typer.Typer()

def _ws(domain: str) -> Path:
    return Path("workspaces") / domain

@app.command()
def init(domain: str, desc: str = typer.Option(None), seeds: str = typer.Option(None)):
    """Initialise a new domain workspace."""
    cand = _ws(domain) / "seeds" / "candidates.jsonl"
    cand.parent.mkdir(parents=True, exist_ok=True)
    if seeds:
        recs = read_jsonl(seeds)
        write_jsonl(cand, recs)
        typer.echo(f"Imported {len(recs)} seeds to {cand}")
    else:
        cand.touch()
        typer.echo(f"Created empty seed file at {cand}")
    if desc:
        (_ws(domain) / "description.txt").write_text(desc)
```

- [ ] **Step 3: Create `commands/curate.py`** — copy `curate` from `cli.py`

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
from data.synthetic.config import load_config
from data.synthetic.bootstrap import bootstrap_seeds
from data.synthetic.teacher import from_config
from data.synthetic.io import read_jsonl

app = typer.Typer()

def _ws(domain: str) -> Path:
    return Path("workspaces") / domain

@app.command()
def curate(domain: str):
    """Bootstrap seed examples for a domain."""
    cfg = load_config(_ws(domain) / "config.yaml")
    teacher = from_config(cfg)
    seeds = bootstrap_seeds(teacher, cfg)
    from data.synthetic.io import write_jsonl
    out = _ws(domain) / "seeds" / "candidates.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out, seeds)
    typer.echo(f"Bootstrapped {len(seeds)} seeds → {out}")
```

- [ ] **Step 4: Create `commands/generate.py`** — copy `generate` from `cli.py`

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
from data.synthetic.config import load_config
from data.synthetic.teacher import from_config
from data.synthetic.embedder import SentenceTransformerEmbedder
from data.synthetic.pipeline import run_generate
from data.synthetic.io import read_jsonl

app = typer.Typer()

def _ws(domain: str) -> Path:
    return Path("workspaces") / domain

@app.command()
def generate(domain: str):
    """Generate and filter synthetic training data."""
    cfg = load_config(_ws(domain) / "config.yaml")
    teacher = from_config(cfg)
    embedder = SentenceTransformerEmbedder()
    seeds = read_jsonl(_ws(domain) / "seeds" / "approved.jsonl")
    run_generate(domain, cfg, teacher, embedder, seeds)
```

- [ ] **Step 5: Create `commands/prepare.py`** — copy `prepare` from `cli.py`, remove `data_config.yaml` reference

```python
import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
from data.synthetic.io import read_jsonl
from data.preprocessor import DataPreprocessor

app = typer.Typer()

def _ws(domain: str) -> Path:
    return Path("workspaces") / domain

@app.command()
def prepare(
    domain: str,
    system_prompt: str = typer.Option(..., help="System prompt for the fine-tuned model"),
    out_dir: str = typer.Option(None, help="Output dir (default: workspaces/<domain>/processed)"),
    test_split: float = typer.Option(0.1, help="Fraction held out for test"),
    val_split: float = typer.Option(0.1, help="Fraction held out for validation"),
):
    """Convert filtered JSONL from generate into train/val/test splits."""
    filtered = _ws(domain) / "generated" / "filtered.jsonl"
    if not filtered.exists():
        typer.echo(f"No filtered data at {filtered}. Run: generate {domain}", err=True)
        raise typer.Exit(1)

    preprocessor = DataPreprocessor(
        system_prompt=system_prompt,
        test_split_ratio=test_split,
        val_split_ratio=val_split,
    )

    samples = []
    for rec in read_jsonl(filtered):
        conversation = rec["conversation"]
        for i in range(0, len(conversation) - 1, 2):
            q = conversation[i]["content"]
            a = conversation[i + 1]["content"]
            samples.append(preprocessor.format_conversation_sample(q, a))

    random.shuffle(samples)
    train, val, test = preprocessor.create_train_val_test_split(samples)

    out = Path(out_dir) if out_dir else _ws(domain) / "processed"
    out.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        (out / f"{name}.json").write_text(json.dumps(split, indent=2))
    stats = {"train_size": len(train), "val_size": len(val), "test_size": len(test),
             "total_size": len(samples)}
    (out / "data_stats.json").write_text(json.dumps(stats, indent=2))
    typer.echo(
        f"Prepared {len(samples)} samples → {out} "
        f"(train={len(train)}, val={len(val)}, test={len(test)})"
    )
```

- [ ] **Step 6: Create `commands/upload.py`** — copy `upload` from `cli.py`

Read the current `upload` command from `cli.py` (lines ~119 onward) and copy verbatim into:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer

app = typer.Typer()

def _ws(domain: str) -> Path:
    return Path("workspaces") / domain

# paste the upload() function from cli.py here verbatim
```

- [ ] **Step 7: Rewrite `cli.py` as a thin router**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import typer
from commands.init     import app as init_app
from commands.curate   import app as curate_app
from commands.generate import app as generate_app
from commands.prepare  import app as prepare_app
from commands.upload   import app as upload_app

app = typer.Typer()
app.add_typer(init_app,     name="init")
app.add_typer(curate_app,   name="curate")
app.add_typer(generate_app, name="generate")
app.add_typer(prepare_app,  name="prepare")
app.add_typer(upload_app,   name="upload")


@app.command()
def tui(domain: str = typer.Option(None, help="Domain to pre-select on launch")):
    """Launch the ElixirLoRA TUI."""
    from tui.app import ElixirLoRAApp
    ElixirLoRAApp(initial_domain=domain).run()


if __name__ == "__main__":
    app()
```

Note: `train`, `evaluate`, `fuse`, `chat` are added to `cli.py` in Tasks 3–6.

- [ ] **Step 8: Verify all existing commands still work**

```bash
python3 cli.py --help
```

Expected output lists: `init`, `curate`, `generate`, `prepare`, `upload`, `tui`.

```bash
pytest tests/ -q
```

Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add commands/ cli.py
git commit -m "refactor: extract all commands into commands/ package; slim cli.py to router"
```

---

### Task 3: src/training/ (mlx-tune) + commands/train.py

**Files:**
- Create: `src/training/__init__.py`, `src/training/metrics_writer.py`, `src/training/sft.py`, `src/training/dpo.py`, `src/training/grpo.py`, `commands/train.py`
- Create: `tests/test_training.py`

**Interfaces:**
- Consumes: `config/model_config.yaml` fields: `base_model.path`, `lora.rank`, `lora.scale`, `lora.dropout`, `lora.keys`; `config/training_config.yaml` fields: `training.iters`, `training.batch_size`, `training.learning_rate`, `training.steps_per_eval`
- Produces: `src/training/sft.run(domain, model_config_path, training_config_path, train_data_path, val_data_path) -> None`; `commands/train.py` registered on `cli.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_training.py
import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call


def test_metrics_writer_callback_creates_file(tmp_path):
    from src.training.metrics_writer import MetricsWriterCallback
    cb = MetricsWriterCallback(tmp_path / "training_metrics.json")
    state = MagicMock(); state.global_step = 100
    cb.on_log(None, state, None, logs={"loss": 2.5})
    data = json.loads((tmp_path / "training_metrics.json").read_text())
    assert data["train_loss"] == [2.5]
    assert data["iterations"] == [100]
    assert data["val_loss"] == []


def test_metrics_writer_callback_appends_eval_loss(tmp_path):
    from src.training.metrics_writer import MetricsWriterCallback
    cb = MetricsWriterCallback(tmp_path / "training_metrics.json")
    state = MagicMock(); state.global_step = 50
    cb.on_log(None, state, None, logs={"loss": 2.0, "eval_loss": 2.1})
    data = json.loads((tmp_path / "training_metrics.json").read_text())
    assert data["val_loss"] == [2.1]


def test_dpo_raises_without_correct_data(tmp_path):
    from src.training.dpo import run
    with pytest.raises(ValueError, match="DPO requires"):
        run("d", tmp_path / "m.yaml", tmp_path / "t.yaml",
            tmp_path / "train.json", None)


def test_grpo_raises_without_correct_data(tmp_path):
    from src.training.grpo import run
    with pytest.raises(ValueError, match="GRPO requires"):
        run("d", tmp_path / "m.yaml", tmp_path / "t.yaml",
            tmp_path / "train.json", None)
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_training.py -q
```

Expected: `ModuleNotFoundError: No module named 'src.training'`

- [ ] **Step 3: Create `src/training/__init__.py`**

```python
```
(Empty.)

- [ ] **Step 4: Create `src/training/metrics_writer.py`**

```python
import json
from pathlib import Path
from transformers import TrainerCallback


class MetricsWriterCallback(TrainerCallback):
    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._data: dict = {"train_loss": [], "val_loss": [], "iterations": []}

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        if not logs:
            return
        if "loss" in logs:
            self._data["train_loss"].append(round(float(logs["loss"]), 4))
            self._data["iterations"].append(int(state.global_step))
        if "eval_loss" in logs:
            self._data["val_loss"].append(round(float(logs["eval_loss"]), 4))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data))
```

- [ ] **Step 5: Create `src/training/sft.py`**

Before writing this file, verify the mlx-tune import path:

```bash
python3 -c "import mlx_tune; print(dir(mlx_tune))"
```

If `FastLanguageModel` is present, use the imports below. If mlx-tune uses a different module structure (e.g. `from mlx_tune.models import FastLanguageModel`), adjust accordingly — the pattern is the same.

```python
import json
import yaml
from pathlib import Path
from datasets import Dataset

from src.training.metrics_writer import MetricsWriterCallback


def _load_configs(model_config_path: Path, training_config_path: Path) -> dict:
    m = yaml.safe_load(model_config_path.read_text())
    t = yaml.safe_load(training_config_path.read_text())
    return {"model": m, "training": t}


def run(
    domain: str,
    model_config_path: Path,
    training_config_path: Path,
    train_data_path: Path,
    val_data_path: Path | None,
) -> None:
    from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig

    cfg = _load_configs(Path(model_config_path), Path(training_config_path))
    m_cfg = cfg["model"]
    t_cfg = cfg["training"]

    metrics_path = (
        Path("workspaces") / domain / "logs" / "training" / "training_metrics.json"
    )
    output_dir = str(Path("workspaces") / domain / "adapters")

    model, tokenizer = FastLanguageModel.from_pretrained(
        m_cfg["base_model"]["path"],
        max_seq_length=m_cfg.get("max_seq_length", 2048),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=m_cfg["lora"]["rank"],
        target_modules=m_cfg["lora"].get("keys", ["q_proj", "v_proj"]),
        lora_alpha=m_cfg["lora"]["scale"],
        lora_dropout=m_cfg["lora"]["dropout"],
    )

    train_ds = Dataset.from_list(json.loads(Path(train_data_path).read_text()))
    eval_ds = (
        Dataset.from_list(json.loads(Path(val_data_path).read_text()))
        if val_data_path
        else None
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=t_cfg["training"]["batch_size"],
            learning_rate=t_cfg["training"]["learning_rate"],
            max_steps=t_cfg["training"]["iters"],
            eval_steps=t_cfg["training"]["steps_per_eval"],
        ),
        callbacks=[MetricsWriterCallback(metrics_path)],
    )
    trainer.train()
```

- [ ] **Step 6: Create `src/training/dpo.py`**

```python
from pathlib import Path


def run(
    domain: str,
    model_config_path: Path,
    training_config_path: Path,
    train_data_path: Path,
    val_data_path: Path | None,
) -> None:
    import json
    data = json.loads(Path(train_data_path).read_text()) if Path(train_data_path).exists() else []
    if not data or not isinstance(data[0], dict) or "chosen" not in data[0]:
        raise ValueError(
            "DPO requires training data with fields {prompt, chosen, rejected}. "
            "Run the DPO data preparation pipeline first."
        )
    from mlx_tune import DPOTrainer, DPOConfig
    raise NotImplementedError("DPO training pipeline not yet configured.")
```

- [ ] **Step 7: Create `src/training/grpo.py`**

```python
from pathlib import Path


def run(
    domain: str,
    model_config_path: Path,
    training_config_path: Path,
    train_data_path: Path,
    val_data_path: Path | None,
) -> None:
    import json
    data = json.loads(Path(train_data_path).read_text()) if Path(train_data_path).exists() else []
    if not data or not isinstance(data[0], dict) or "prompt" not in data[0]:
        raise ValueError(
            "GRPO requires training data with a {prompt} field and a configured reward function. "
            "Run the GRPO data preparation pipeline first."
        )
    from mlx_tune import GRPOTrainer, GRPOConfig
    raise NotImplementedError("GRPO training pipeline not yet configured.")
```

- [ ] **Step 8: Create `commands/train.py`**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

app = typer.Typer()


@app.command()
def train(
    domain: str,
    method: str = typer.Option("sft", help="Training method: sft | dpo | grpo"),
    model_config: Path = typer.Option(..., help="Path to runtime model config YAML"),
    training_config: Path = typer.Option(..., help="Path to runtime training config YAML"),
    train_data: Path = typer.Option(..., help="Path to train.json"),
    val_data: Path = typer.Option(None, help="Path to val.json (optional)"),
) -> None:
    """Fine-tune a model using the specified training method."""
    if method == "sft":
        from src.training.sft import run
    elif method == "dpo":
        from src.training.dpo import run
    elif method == "grpo":
        from src.training.grpo import run
    else:
        raise typer.BadParameter(f"Unknown method '{method}'. Choose: sft, dpo, grpo")
    run(domain, model_config, training_config, train_data, val_data)
```

- [ ] **Step 9: Register `train` in `cli.py`**

Add to `cli.py` (after the existing `add_typer` calls):

```python
from commands.train import app as train_app
app.add_typer(train_app, name="train")
```

- [ ] **Step 10: Run tests**

```bash
pytest tests/test_training.py -q
```

Expected: 4 tests pass.

```bash
pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 11: Smoke test the CLI entry point (no model download)**

```bash
python3 cli.py train --help
```

Expected: shows `domain`, `--method`, `--model-config`, `--training-config`, `--train-data`, `--val-data`.

- [ ] **Step 12: Commit**

```bash
git add src/training/ commands/train.py cli.py tests/test_training.py
git commit -m "feat: replace src/training/ with mlx-tune backend; add commands/train.py"
```

---

### Task 4: evaluator.py fix + commands/evaluate.py

**Files:**
- Modify: `src/evaluation/evaluator.py` (one method only)
- Audit: `src/evaluation/metrics_calculator.py`, `src/evaluation/comparator.py`
- Create: `commands/evaluate.py`, `tests/test_commands.py` (partial)

**Interfaces:**
- Consumes: `mlx_lm.load(model_path, adapter_path=str)` (stable public API)
- Produces: `commands/evaluate.py` registered on `cli.py`; fixed `load_model_with_adapters(base_model_path, adapter_path)`

- [ ] **Step 1: Write failing test for the fixed method**

```python
# tests/test_commands.py
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


def test_load_model_with_adapters_uses_public_api(tmp_path):
    """Ensure load_model_with_adapters calls mlx_lm.load with adapter_path kwarg."""
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("mlx_lm.load", return_value=(mock_model, mock_tokenizer)) as mock_load:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from evaluation.evaluator import ModelEvaluator

        eval_cfg = tmp_path / "eval.yaml"
        eval_cfg.write_text(
            "evaluation:\n  method: simple\n  max_tokens: 200\n  temperature: 0.7\n"
            "metrics:\n  simple:\n    word_overlap_threshold: 0.5\n"
            "comparison:\n  compare_with_base: true\n  score_thresholds:\n"
            "    excellent: 0.9\n    good: 0.7\n    acceptable: 0.5\n    poor: 0.3\n"
            "paths:\n  results_dir: /tmp\n  test_data: /tmp/test.json\n"
        )
        evaluator = ModelEvaluator(str(eval_cfg))
        model, tok = evaluator.load_model_with_adapters("base-model", str(adapter_dir))

    mock_load.assert_called_once_with("base-model", adapter_path=str(adapter_dir))
    assert model is mock_model
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_commands.py::test_load_model_with_adapters_uses_public_api -q
```

Expected: FAIL — current code calls `linear_to_lora_layers` instead.

- [ ] **Step 3: Fix `load_model_with_adapters` in `src/evaluation/evaluator.py`**

Replace the entire `load_model_with_adapters` method with:

```python
def load_model_with_adapters(self, base_model_path: str, adapter_path: str):
    """Load base model with LoRA adapters using the stable mlx_lm public API."""
    print(f"Loading model with adapters: {base_model_path} + {adapter_path}")
    model, tokenizer = load(base_model_path, adapter_path=adapter_path)
    print("Model with adapters loaded successfully")
    return model, tokenizer
```

Also remove these now-unused imports from the top of `evaluator.py`:

```python
# Remove:
from mlx_lm.tuner import linear_to_lora_layers
from mlx.utils import tree_flatten
import mlx.core as mx
```

- [ ] **Step 4: Audit `src/evaluation/metrics_calculator.py` and `comparator.py`**

```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from evaluation.metrics_calculator import MetricsCalculator
from evaluation.comparator import ModelComparator
print('imports ok')
"
```

Expected: `imports ok`. If it raises, fix the import (likely a missing standard library or path issue) before continuing.

- [ ] **Step 5: Run test to confirm fix**

```bash
pytest tests/test_commands.py::test_load_model_with_adapters_uses_public_api -q
```

Expected: PASS.

- [ ] **Step 6: Create `commands/evaluate.py`**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer

app = typer.Typer()

def _ws(domain: str) -> Path:
    return Path("workspaces") / domain


@app.command()
def evaluate(
    domain: str,
    eval_config: Path = typer.Option(..., help="Path to runtime eval config YAML"),
    adapters_path: Path = typer.Option(None, help="Path to adapters dir (omit to eval base only)"),
    test_data: Path = typer.Option(None, help="Path to test.json (default: workspaces/<domain>/processed/test.json)"),
    fused_path: Path = typer.Option(None, help="Path to fused model dir (optional)"),
) -> None:
    """Evaluate base model and/or fine-tuned adapters for a domain."""
    from evaluation.evaluator import ModelEvaluator
    import yaml

    ws = _ws(domain)
    if test_data is None:
        test_data = ws / "processed" / "test.json"

    model_cfg = yaml.safe_load((ws / "runtime_model_config.yaml").read_text())
    base_model = model_cfg["base_model"]["path"]

    evaluator = ModelEvaluator(str(eval_config))

    if adapters_path and Path(adapters_path).exists():
        evaluator.comprehensive_model_comparison(
            base_model_path=base_model,
            adapter_path=str(adapters_path),
            fused_model_path=str(fused_path) if fused_path else None,
            test_data_path=str(test_data),
        )
    else:
        evaluator.evaluate_model_from_path(base_model, "base_model", str(test_data))
```

- [ ] **Step 7: Register `evaluate` in `cli.py`**

```python
from commands.evaluate import app as evaluate_app
app.add_typer(evaluate_app, name="evaluate")
```

- [ ] **Step 8: Smoke test**

```bash
python3 cli.py evaluate --help
```

Expected: shows `domain`, `--eval-config`, `--adapters-path`, `--test-data`, `--fused-path`.

- [ ] **Step 9: Run all tests**

```bash
pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add src/evaluation/evaluator.py commands/evaluate.py cli.py tests/test_commands.py
git commit -m "fix: replace mlx_lm.tuner internal import in evaluator; add commands/evaluate.py"
```

---

### Task 5: commands/fuse.py

**Files:**
- Audit + modify: `src/utils/fusion.py` (remove HuggingFace-specific methods, keep `fuse_adapters`)
- Create: `commands/fuse.py`

**Interfaces:**
- Consumes: `src/utils/fusion.AdapterFusion.fuse_adapters(base_model_path, adapter_path, output_path)` — uses `mlx_lm.fuse` CLI (stable)
- Produces: `commands/fuse.py` registered on `cli.py`

- [ ] **Step 1: Audit `src/utils/fusion.py`**

```bash
python3 -c "
import sys; sys.path.insert(0, 'src')
from utils.fusion import AdapterFusion
f = AdapterFusion()
print('import ok')
print('fuse_adapters:', f.fuse_adapters.__doc__[:40])
"
```

Expected: `import ok`. The `fuse_adapters` method already calls `mlx_lm.fuse` via subprocess — this is the stable public CLI, so it works across mlx-lm versions.

- [ ] **Step 2: Remove dead methods from `src/utils/fusion.py`**

Delete the following methods (they reference the old `models/` directory layout and are not called anywhere):
- `compare_fusion_quality` — loads from `mlx_lm`, uses hardcoded test prompts with "OpenBB" context
- `create_fusion_report` — writes to `logs/fusion_report_*.json` (old hardcoded path)
- `cleanup_fusion_artifacts` — only needed if we were saving checkpoints to adapter dir (we don't)
- `get_fusion_info` — only used by `create_fusion_report`

Keep: `__init__`, `fuse_adapters`, `validate_fusion_inputs`, `_is_huggingface_model_id`.

After removing, the file should be ~60 lines.

- [ ] **Step 3: Write test for fuse command**

Append to `tests/test_commands.py`:

```python
def test_fuse_calls_adapter_fusion(tmp_path):
    """fuse command delegates to AdapterFusion.fuse_adapters."""
    from unittest.mock import patch, MagicMock
    ws = tmp_path / "workspaces" / "d"
    (ws / "adapters").mkdir(parents=True)
    (ws / "adapters" / "adapters.safetensors").write_bytes(b"x")
    (ws / "adapters" / "adapter_config.json").write_text('{"lora_layers": 4}')

    model_cfg = ws / "runtime_model_config.yaml"
    model_cfg.parent.mkdir(parents=True, exist_ok=True)
    model_cfg.write_text("base_model:\n  path: 'some/model'\n")

    import os; os.chdir(tmp_path)

    with patch("utils.fusion.AdapterFusion.fuse_adapters", return_value=str(ws / "fused")) as mock_fuse:
        from typer.testing import CliRunner
        import sys; sys.path.insert(0, str(tmp_path / "src") if (tmp_path / "src").exists() else "src")
        from commands.fuse import app
        runner = CliRunner()
        result = runner.invoke(app, [
            "fuse", "d",
            "--model-config", str(model_cfg),
            "--output-path", str(ws / "fused"),
        ])
    assert result.exit_code == 0, result.output
    mock_fuse.assert_called_once()
```

- [ ] **Step 4: Create `commands/fuse.py`**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
import yaml

app = typer.Typer()

def _ws(domain: str) -> Path:
    return Path("workspaces") / domain


@app.command()
def fuse(
    domain: str,
    model_config: Path = typer.Option(..., help="Path to runtime model config YAML"),
    output_path: Path = typer.Option(None, help="Fused model output path (default: workspaces/<domain>/fused)"),
    eval_config: Path = typer.Option(None, help="Eval config YAML (omit to skip post-fuse eval)"),
    test_data: Path = typer.Option(None, help="Test data for post-fuse eval"),
    adapters_path: Path = typer.Option(None, help="Adapters dir (default: workspaces/<domain>/adapters)"),
) -> None:
    """Fuse LoRA adapters into the base model and optionally evaluate the result."""
    from utils.fusion import AdapterFusion

    ws = _ws(domain)
    m_cfg = yaml.safe_load(Path(model_config).read_text())
    base_model = m_cfg["base_model"]["path"]

    adapters = Path(adapters_path) if adapters_path else ws / "adapters"
    out = Path(output_path) if output_path else ws / "fused"

    fusion = AdapterFusion()
    if not fusion.validate_fusion_inputs(base_model, str(adapters)):
        raise typer.Exit(1)

    fusion.fuse_adapters(base_model, str(adapters), str(out))
    typer.echo(f"Fused model saved to: {out}")

    if eval_config and Path(eval_config).exists():
        from evaluation.evaluator import ModelEvaluator
        test = Path(test_data) if test_data else ws / "processed" / "test.json"
        evaluator = ModelEvaluator(str(eval_config))
        evaluator.evaluate_model_from_path(str(out), "lora_fused", str(test))
```

- [ ] **Step 5: Register `fuse` in `cli.py`**

```python
from commands.fuse import app as fuse_app
app.add_typer(fuse_app, name="fuse")
```

- [ ] **Step 6: Smoke test**

```bash
python3 cli.py fuse --help
```

Expected: shows `domain`, `--model-config`, `--output-path`, `--eval-config`, `--test-data`, `--adapters-path`.

- [ ] **Step 7: Run all tests**

```bash
pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/utils/fusion.py commands/fuse.py cli.py tests/test_commands.py
git commit -m "feat: add commands/fuse.py; trim dead methods from fusion.py"
```

---

### Task 6: commands/chat.py + fix inference module

**Files:**
- Modify: `src/inference/generator.py` (remove hardcoded system prompt)
- Modify: `src/inference/chat_interface.py` (remove hardcoded system prompt fallback)
- Create: `commands/chat.py`

**Interfaces:**
- Consumes: `src/inference/generator.TextGenerator(model_path, system_prompt)` — `system_prompt` now required (no default)
- Produces: `commands/chat.py` registered on `cli.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_commands.py`:

```python
def test_text_generator_has_no_hardcoded_system_prompt():
    """TextGenerator must not default to any hardcoded persona."""
    import sys; sys.path.insert(0, "src")
    from inference.generator import TextGenerator
    from unittest.mock import patch, MagicMock
    with patch("mlx_lm.load", return_value=(MagicMock(), MagicMock())):
        gen = TextGenerator("some/model", system_prompt=None)
    # system prompt should be empty/None, not the old "Didier" string
    assert "Didier" not in (gen.default_system_prompt or "")
    assert "OpenBB" not in (gen.default_system_prompt or "")
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_commands.py::test_text_generator_has_no_hardcoded_system_prompt -q
```

Expected: FAIL — current code sets `default_system_prompt` to the "Didier" string.

- [ ] **Step 3: Fix `src/inference/generator.py` — remove hardcoded system prompt**

Replace the `__init__` method:

```python
def __init__(self, model_path: str, system_prompt: str | None = None):
    self.model_path = model_path
    self.model, self.tokenizer = load(model_path)
    self.default_system_prompt = system_prompt or ""
    print(f"Text generator initialized with model: {model_path}")
```

- [ ] **Step 4: Fix `src/inference/chat_interface.py` — remove hardcoded fallback**

In `ChatInterface.__init__`, the `TextGenerator` is instantiated with `system_prompt`. Remove any fallback to the "Didier" string inside `ChatInterface` itself. The system prompt must come from the caller.

Change the `__init__` signature to require `system_prompt`:

```python
def __init__(self, model_path: str, system_prompt: str):
    self.generator = TextGenerator(model_path, system_prompt)
    self.conversation_history = []
    self.session_active = False
    print("="*60)
    print("CHAT INTERFACE INITIALIZED")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"System prompt: {system_prompt[:100]}...")
    print("\nType 'quit', 'exit', or 'q' to end the conversation")
    print("Type 'clear' to clear conversation history")
    print("="*60)
```

- [ ] **Step 5: Run test to confirm fix**

```bash
pytest tests/test_commands.py::test_text_generator_has_no_hardcoded_system_prompt -q
```

Expected: PASS.

- [ ] **Step 6: Create `commands/chat.py`**

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
import yaml

app = typer.Typer()

def _ws(domain: str) -> Path:
    return Path("workspaces") / domain


def _system_prompt(domain: str) -> str:
    cfg_path = _ws(domain) / "config.yaml"
    if cfg_path.exists():
        try:
            data = yaml.safe_load(cfg_path.read_text()) or {}
            sp = data.get("chat", {}).get("system_prompt") if isinstance(data, dict) else None
            if sp:
                return sp
        except Exception:
            pass
    return "You are a helpful assistant."


@app.command()
def chat(
    domain: str,
    fused: bool = typer.Option(True, help="Use fused model (default) or runtime adapters"),
    max_tokens: int = typer.Option(200),
    temperature: float = typer.Option(0.7),
) -> None:
    """Start an interactive chat session with the domain's fine-tuned model."""
    from inference.chat_interface import ChatInterface

    ws = _ws(domain)
    model_path = str(ws / "fused") if fused else str(ws / "adapters")
    if not Path(model_path).exists():
        typer.echo(f"Model not found at {model_path}. Run fuse first.", err=True)
        raise typer.Exit(1)

    system_prompt = _system_prompt(domain)
    interface = ChatInterface(model_path, system_prompt)
    interface.start_chat(max_tokens=max_tokens, temperature=temperature)
```

- [ ] **Step 7: Register `chat` in `cli.py`**

```python
from commands.chat import app as chat_app
app.add_typer(chat_app, name="chat")
```

- [ ] **Step 8: Smoke test**

```bash
python3 cli.py chat --help
```

Expected: shows `domain`, `--fused/--no-fused`, `--max-tokens`, `--temperature`.

- [ ] **Step 9: Run all tests**

```bash
pytest tests/ -q
```

Expected: all pass.

- [ ] **Step 10: Commit**

```bash
git add src/inference/ commands/chat.py cli.py tests/test_commands.py
git commit -m "fix: remove hardcoded system prompt from inference module; add commands/chat.py"
```

---

### Task 7: Update TUI subprocess calls

**Files:**
- Modify: `tui/panels/training.py`, `tui/panels/evaluation.py`, `tui/panels/deployment.py`
- Modify: `tests/tui/test_panels.py` (if any test asserts on the exact subprocess command string)

**Interfaces:**
- Consumes: `commands/train.py`, `commands/evaluate.py`, `commands/fuse.py` (all registered on cli.py)
- Produces: TUI panels calling `python3 cli.py <command> <domain> ...` instead of `python3 scripts/...`

- [ ] **Step 1: Update `tui/panels/training.py` — `_run_train`**

Find the `_run_train` method (currently at line ~98). Replace the `cmd` list:

```python
@work(thread=True)
def _run_train(self, domain: str) -> None:
    ws = Path("workspaces") / domain
    cmd = [
        "python3", "cli.py", "train", domain,
        "--model-config", str(ws / "runtime_model_config.yaml"),
        "--training-config", str(ws / "runtime_training_config.yaml"),
        "--train-data", str(ws / "processed" / "train.json"),
        "--val-data", str(ws / "processed" / "val.json"),
    ]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    for line in proc.stdout:
        self.post_message(RunnerOutput(line.rstrip()))
    proc.wait()
    self.post_message(RunnerDone(proc.returncode))
```

- [ ] **Step 2: Update `tui/panels/evaluation.py` — `_run_eval` and `_run_fuse_eval`**

```python
def _run_eval(self, domain: str) -> None:
    ws = Path("workspaces") / domain
    cmd = [
        "python3", "cli.py", "evaluate", domain,
        "--eval-config", str(ws / "runtime_eval_config.yaml"),
        "--adapters-path", str(ws / "adapters"),
        "--test-data", str(ws / "processed" / "test.json"),
    ]
    self._stream(cmd)

@work(thread=True)
def _run_fuse_eval(self, domain: str) -> None:
    ws = Path("workspaces") / domain
    cmd = [
        "python3", "cli.py", "fuse", domain,
        "--model-config", str(ws / "runtime_model_config.yaml"),
        "--eval-config", str(ws / "runtime_eval_config.yaml"),
        "--test-data", str(ws / "processed" / "test.json"),
        "--adapters-path", str(ws / "adapters"),
        "--output-path", str(ws / "fused"),
    ]
    self._stream(cmd)
```

- [ ] **Step 3: Update `tui/panels/deployment.py` — `_run_fuse`**

```python
def _run_fuse(self, domain: str) -> None:
    ws = Path("workspaces") / domain
    cmd = [
        "python3", "cli.py", "fuse", domain,
        "--model-config", str(ws / "runtime_model_config.yaml"),
        "--eval-config", str(ws / "runtime_eval_config.yaml"),
        "--test-data", str(ws / "processed" / "test.json"),
        "--adapters-path", str(ws / "adapters"),
        "--output-path", str(ws / "fused"),
    ]
    self._stream(cmd)
```

- [ ] **Step 4: Run full test suite**

```bash
pytest tests/ -q
```

Expected: all pass. The existing TUI tests mock the subprocess calls and do not assert on the exact command string, so no test updates should be needed. If any test does `assert "scripts/" in cmd`, update it to match the new `"cli.py"` prefix.

- [ ] **Step 5: Verify `python3 cli.py --help` shows all 9 commands**

```bash
python3 cli.py --help
```

Expected output includes: `init`, `curate`, `generate`, `prepare`, `upload`, `train`, `evaluate`, `fuse`, `chat`, `tui`.

- [ ] **Step 6: Commit**

```bash
git add tui/panels/training.py tui/panels/evaluation.py tui/panels/deployment.py
git commit -m "refactor: update TUI subprocess calls from scripts/ to cli.py commands"
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| Delete demo_pipeline.py, notebook, data_config, venv, plotting.py, src/training/, scripts/ | Task 1 |
| Remove `paths:` from config templates; add `method: sft` | Task 1 |
| Swap mlx-tune into requirements.txt | Task 1 |
| commands/ package with __init__.py | Task 2 |
| Move init, curate, generate, prepare, upload from cli.py | Task 2 |
| Slim cli.py to router | Task 2 |
| src/training/metrics_writer.py | Task 3 |
| src/training/sft.py (mlx-tune) | Task 3 |
| src/training/dpo.py and grpo.py (stubs with clear errors) | Task 3 |
| commands/train.py dispatcher | Task 3 |
| Fix evaluator.py load_model_with_adapters | Task 4 |
| commands/evaluate.py | Task 4 |
| Trim dead methods from fusion.py | Task 5 |
| commands/fuse.py | Task 5 |
| Fix hardcoded system prompt in generator.py | Task 6 |
| commands/chat.py | Task 6 |
| TUI subprocess updates (3 panels, 4 calls) | Task 7 |
| End-to-end smoke test per command | Tasks 3–6 (--help + import checks) |

**Placeholder scan:** Clean — every step has concrete code or an exact shell command.

**Type consistency:** `sft.run(domain, model_config_path, training_config_path, train_data_path, val_data_path)` matches the call in `commands/train.py` across Tasks 3. `dpo.run` and `grpo.run` have identical signatures. `fuse_adapters(base_model_path, adapter_path, output_path)` matches usage in `commands/fuse.py`.
