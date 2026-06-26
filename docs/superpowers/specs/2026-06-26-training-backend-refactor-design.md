# Training Backend Refactor — Design Spec

## Goal

Replace the broken `src/training/` module and the numbered `scripts/` folder with a clean, maintainable architecture: a `commands/` package backed by mlx-tune (SFT + DPO + GRPO), and a `cli.py` that is the single entry point for all pipeline operations.

## Architecture

### Why this is needed

`src/training/` imports private internals from `mlx_lm.tuner` (`linear_to_lora_layers`, `TrainingArgs`, `train`) that have drifted out of sync with the current mlx-lm API. The same internal import appears in `src/evaluation/evaluator.py`. Neither is callable today.

`scripts/01_*.py` … `scripts/04_*.py` duplicate entry points that already belong in the CLI.

### What gets deleted

| Path | Reason |
|---|---|
| `src/training/trainer.py` | Entirely replaced by mlx-tune |
| `src/training/lora_setup.py` | Entirely replaced by mlx-tune |
| `src/training/metrics.py` | Replaced by `MetricsWriterCallback` |
| `src/training/__init__.py` | Module gone |
| `scripts/01_prepare_data.py` | Moved to `commands/prepare.py` |
| `scripts/02_train_model.py` | Moved to `commands/train.py` |
| `scripts/03_evaluate_model.py` | Moved to `commands/evaluate.py` |
| `scripts/04_fuse_and_evaluate.py` | Moved to `commands/fuse.py` |
| `scripts/upload_model_to_hf.py` | Already in `commands/upload.py` (from cli.py) |
| `scripts/interactive_chat.py` | Moved to `commands/chat.py` |

### New file structure

```
cli.py                          # Entry point only (~10 lines)
commands/
  __init__.py
  init.py                       # domain init + seed import (moved from cli.py)
  prepare.py                    # data preparation
  train.py                      # train dispatcher: --method sft|dpo|grpo
  evaluate.py                   # evaluation
  fuse.py                       # adapter fusion + evaluation
  upload.py                     # HuggingFace upload (moved from cli.py)
  chat.py                       # interactive chat
src/
  data/                         # unchanged
  training/                     # NEW: clean mlx-tune wrappers
    __init__.py
    sft.py                      # FastLanguageModel + SFTTrainer
    dpo.py                      # DPO trainer (stub until data pipeline ready)
    grpo.py                     # GRPO trainer (stub until data pipeline ready)
    metrics_writer.py           # TrainerCallback → training_metrics.json
  evaluation/                   # one method fix only (see below)
  inference/                    # unchanged
tui/                            # subprocess call strings updated only
```

### `cli.py` — entry point only

```python
import typer
from commands.init    import app as init_app
from commands.prepare import app as prepare_app
from commands.train   import app as train_app
from commands.evaluate import app as evaluate_app
from commands.fuse    import app as fuse_app
from commands.upload  import app as upload_app
from commands.chat    import app as chat_app

app = typer.Typer()
app.add_typer(init_app,     name="init")
app.add_typer(prepare_app,  name="prepare")
app.add_typer(train_app,    name="train")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(fuse_app,     name="fuse")
app.add_typer(upload_app,   name="upload")
app.add_typer(chat_app,     name="chat")

if __name__ == "__main__":
    app()
```

---

## Training backend

### Why mlx-tune

| Requirement | mlx-lm (official) | mlx-tune |
|---|---|---|
| SFT / LoRA | Yes (CLI only) | Yes (Python API) |
| DPO | No | Yes |
| GRPO | No | Yes |
| Stable public API | Yes (`mlx_lm.lora` CLI) | Yes (Unsloth-compatible) |
| Last release | Feb 2026 (v0.31.0) | May 2026 (v0.5.1) |

mlx-tune is the only option that covers all three training methods without custom implementation. It wraps mlx-lm internally, so it absorbs internal API churn. The Unsloth-compatible API (`FastLanguageModel`, `SFTTrainer`) is designed to be stable.

### `commands/train.py`

Reads `method` from `training_config.yaml` (default `sft`). Accepts the same logical arguments as the old script so TUI subprocess calls require only a path change.

```python
@app.command()
def train(
    domain: str,
    method: str = typer.Option("sft"),
    model_config: Path = typer.Option(...),
    training_config: Path = typer.Option(...),
    train_data: Path = typer.Option(...),
    val_data: Path = typer.Option(None),
):
    if method == "sft":
        from src.training.sft import run
    elif method == "dpo":
        from src.training.dpo import run
    elif method == "grpo":
        from src.training.grpo import run
    else:
        raise typer.BadParameter(f"Unknown method: {method}")
    run(domain, model_config, training_config, train_data, val_data)
```

### `src/training/sft.py`

```python
from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig
from datasets import Dataset
from src.training.metrics_writer import MetricsWriterCallback

def run(domain, model_config_path, training_config_path, train_data_path, val_data_path):
    cfg = _load_configs(model_config_path, training_config_path)
    metrics_path = Path("workspaces") / domain / "logs" / "training" / "training_metrics.json"

    model, tokenizer = FastLanguageModel.from_pretrained(
        cfg["base_model"]["path"],
        max_seq_length=cfg.get("max_seq_length", 2048),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["rank"],
        target_modules=cfg["lora"].get("keys", ["q_proj", "v_proj"]),
        lora_alpha=cfg["lora"]["scale"],
        lora_dropout=cfg["lora"]["dropout"],
    )

    train_ds = Dataset.from_list(json.loads(train_data_path.read_text()))
    eval_ds  = Dataset.from_list(json.loads(val_data_path.read_text())) if val_data_path else None

    output_dir = str(Path("workspaces") / domain / "adapters")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=cfg["training"]["batch_size"],
            learning_rate=cfg["training"]["learning_rate"],
            max_steps=cfg["training"]["iters"],
            eval_steps=cfg["training"]["steps_per_eval"],
        ),
        callbacks=[MetricsWriterCallback(metrics_path)],
    )
    trainer.train()
```

### `src/training/dpo.py` and `grpo.py`

Same `run()` signature. Use `DPOTrainer`/`GRPOTrainer` from mlx-tune. Both files are stubs until the data preparation pipeline for those methods exists — calling them without correctly formatted data raises a clear `ValueError` with a message explaining the required format. They are NOT empty files; they contain the full trainer setup behind the format check.

### `src/training/metrics_writer.py`

Writes the `training_metrics.json` format the TUI polls every 2 seconds. Appends each eval step's loss; creates the file on first write.

```python
from transformers import TrainerCallback
import json
from pathlib import Path

class MetricsWriterCallback(TrainerCallback):
    def __init__(self, path: Path):
        self._path = path
        self._data = {"train_loss": [], "val_loss": [], "iterations": []}

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        if "loss" in logs:
            self._data["train_loss"].append(round(logs["loss"], 4))
            self._data["iterations"].append(state.global_step)
        if "eval_loss" in logs:
            self._data["val_loss"].append(round(logs["eval_loss"], 4))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._data))
```

---

## Evaluation fix

`src/evaluation/evaluator.py` uses `linear_to_lora_layers` from `mlx_lm.tuner` (same broken internal import) to apply LoRA adapters at inference time. Replace with the public `mlx_lm.load(adapter_path=...)` API:

```python
# Before (broken):
from mlx_lm.tuner import linear_to_lora_layers
model, tokenizer = load(base_model_path)
model.freeze()
linear_to_lora_layers(model, adapter_config["lora_layers"], adapter_config["lora_parameters"])
model.load_weights(str(adapter_file), strict=False)

# After (stable):
model, tokenizer = load(base_model_path, adapter_path=adapter_path)
```

Only `load_model_with_adapters()` in `evaluator.py` changes. All other logic (metrics, comparison, saving) is untouched. The `adapter_config.json` file is no longer read by the evaluator — mlx-lm's `load()` reads it internally.

---

## TUI subprocess changes

Only the command strings in the TUI panel workers change. No logic changes.

| Panel | Before | After |
|---|---|---|
| TrainingPanel | `python3 scripts/02_train_model.py --model-config ... --train-data ...` | `python3 cli.py train <domain> --method sft --model-config ... --train-data ...` |
| EvaluationPanel | `python3 scripts/03_evaluate_model.py ...` | `python3 cli.py evaluate <domain> ...` |
| DeploymentPanel (fuse) | `python3 scripts/04_fuse_and_evaluate.py ...` | `python3 cli.py fuse <domain> ...` |

---

## Data pipeline

### SFT (in scope)

Current `processed/train.json` is a JSON array. mlx-tune consumes HuggingFace `Dataset` objects. `sft.py` converts in-memory with `Dataset.from_list(json.loads(...))` — no changes to `commands/prepare.py` or the data format on disk.

### DPO and GRPO (out of scope for this refactor)

DPO requires `{"prompt": str, "chosen": str, "rejected": str}` triples. GRPO requires prompts and a reward function. These need new data preparation flows. The training modules will exist and be callable, but `commands/prepare.py` does not yet produce their data formats. This is a future slice.

---

## Dependency changes

`requirements.txt`:
- Add: `mlx-tune>=0.5.0`
- Remove: `mlx-lm>=0.15.0` (mlx-tune pins and installs it as a dependency)

---

## Out of scope

- Data preparation logic changes (only entry point moves from `scripts/` to `commands/`)
- Evaluation metrics logic
- TUI panel logic
- `src/inference/`, `src/data/`
- DPO and GRPO data preparation pipelines
