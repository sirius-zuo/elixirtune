# ElixirLoRA — TUI Dashboard (Design Spec)

**Version:** 1.0
**Date:** 2026-06-25
**Slice:** TUI Dashboard — second vertical slice of ElixirLoRA
**Status:** Approved design — ready for implementation planning

---

## 1. Scope & Integration Boundary

This spec covers the TUI Dashboard — the top orchestration layer of ElixirLoRA. It wraps every stage of the pipeline into a single interactive terminal application.

### What we build
A Textual-based TUI that:
- Manages multiple domain workspaces from a sidebar
- Provides five pipeline panels per domain: Overview, Synthetic Data, Training, Evaluation, Deployment
- Invokes all pipeline stages as subprocesses, streaming their stdout live
- Offers light inline editing of key config knobs
- Exposes domain status derived from workspace file state

### What we do NOT build
- New pipeline logic (all logic stays in `cli.py` and `scripts/`)
- Per-provider training abstractions (the fork's scripts handle this)
- Web UI or remote dashboard
- Multi-adapter runtime switching (future slice)

### Integration boundary
The TUI calls `python cli.py <command>` and `python scripts/<script>.py` as subprocesses. It reads output files (`logs/`, `models/`, `workspaces/`) to display results. It writes config changes directly to YAML files. No new Python APIs are introduced between the TUI and the pipeline.

---

## 2. Module Layout

Additive to the existing codebase. No modifications to `scripts/`, `src/`, or the fork's files.

```
tui/
  __init__.py
  app.py          # Textual App root; mounts sidebar + content area; registers keybindings
  domain.py       # DomainState dataclass + workspace scanner
  runner.py       # SubprocessRunner worker — streams subprocess stdout into a Log widget
  sidebar.py      # DomainList widget with status dots + [+ New Domain] button
  panels/
    __init__.py
    overview.py   # Status summary + Run Full Pipeline
    synthetic.py  # Init/Curate/Generate/Prepare controls + config form + log
    training.py   # Train button + config form + log + loss sparkline
    evaluation.py # Evaluate button + results table + log
    deployment.py # Fuse button + adapter inventory + Ollama action + log
  widgets/
    __init__.py
    config_form.py  # Reusable inline YAML-knob editor (reads + writes YAML)
    log_view.py     # Log widget wrapper with clear action
```

### Entry point
New `tui` command added to `cli.py`:

```
python cli.py tui [--domain DOMAIN]
```

`--domain` pre-selects a domain on launch. Without it, the sidebar receives focus.

---

## 3. Layout

```
┌─ ElixirLoRA ──────────────────────────────────────────────────────┐
│ DOMAINS           │ Overview  Synth  Training  Eval  Deploy       │
│ ──────────────── │ ──────────────────────────────────────────── │
│ ● code_review    │                                               │
│ ○ architecture   │            active panel content               │
│ ○ sql_agent      │                                               │
│                  │                                               │
│  [+ New Domain]  │                                               │
└──────────────────┴───────────────────────────────────────────────┘
```

- **Left sidebar** (fixed ~20 cols): scrollable domain list, status dots, `[+ New Domain]` button at bottom.
- **Right content area**: tab bar across the top, active panel below. Tabs are per-domain context; switching domain preserves the active tab.
- **Status dots**: `●` deployed/evaluated · `◉` trained/prepared/generated · `○` seeded/empty.

---

## 4. Domain State

State is inferred from workspace file presence — no separate state file.

| State | Determining condition |
|---|---|
| `empty` | workspace dir exists; no `seeds/approved.jsonl` |
| `seeded` | `seeds/approved.jsonl` present |
| `generated` | `generated/filtered.jsonl` present |
| `prepared` | `workspaces/<domain>/processed/train.json` present |
| `trained` | `workspaces/<domain>/adapters/` non-empty |
| `evaluated` | `workspaces/<domain>/evaluation/` contains at least one `*_evaluation.json` |
| `deployed` | `workspaces/<domain>/fused/` non-empty |

States are re-scanned after every subprocess completes so the sidebar stays current. The scanner also runs on app startup and when the user adds a new domain.

### Domain-scoped output paths

The fork's scripts write to global paths by default (`data/processed/`, `models/adapters/`, `logs/`). With multiple domains these collide. The TUI solves this by generating a **per-domain config overlay** at runtime before invoking any fork script:

```
workspaces/<domain>/
  processed/           ← replaces data/processed/
  adapters/            ← replaces models/adapters/
  fused/               ← replaces models/fused/
  logs/training/       ← replaces logs/training/
  logs/evaluation/     ← replaces logs/evaluation/
  runtime_data_config.yaml      ← generated overlay passed to scripts/01
  runtime_model_config.yaml     ← generated overlay passed to scripts/02,04
  runtime_training_config.yaml  ← generated overlay passed to scripts/02
  runtime_eval_config.yaml      ← generated overlay passed to scripts/03,04
```

The `prepare` CLI command already accepts `--system-prompt` and writes to its own output path; it will also receive a `--out-dir` flag pointing to `workspaces/<domain>/processed/`.

All fork script invocations in the TUI use the explicit `--model-config`, `--training-config`, `--eval-config`, `--train-data`, `--val-data`, `--test-data`, `--adapters-path`, `--output-path` flags that the scripts already support — no modification to the scripts needed.

Runtime config overlays are generated **lazily, immediately before each subprocess call** — not on domain selection. The generator reads `config/model_config.yaml` (or `config/training_config.yaml` etc.) as the base, deep-merges the domain-scoped path overrides, and writes the result to `workspaces/<domain>/runtime_*.yaml`.

---

## 5. SubprocessRunner

`runner.py` provides one `SubprocessRunner` Textual worker, shared across panels:

- Spawns the command with `stdout=PIPE, stderr=STDOUT` so all output arrives on one stream.
- Reads lines in a background thread; posts each as a Textual message to the calling panel's `LogView`.
- On exit code ≠ 0: posts an `RunFailed` message; the panel marks its action button red and shows the exit code.
- Only one subprocess per domain may run at a time. Action buttons for the active domain are disabled while a runner is live.
- `Kill` button appears while a subprocess is running; sends `SIGTERM` to the process.

---

## 6. Panel Designs

### 6.1 Overview
Displays a read-only status summary for the selected domain. Pulls numbers directly from workspace files at render time (no caching).

```
Status: generated               Last run: 2026-06-25T14-30
Seeds:     23 approved
Generated: 1847 / 2000 target
Prepared:  train=1478  val=185  test=184
Adapter:   models/adapters/  (not yet trained)

[▶ Run Full Pipeline]
```

**Run Full Pipeline** runs each stage subprocess in sequence:
`init` (if needed) → `generate` → `prepare` → `02_train_model.py` → `03_evaluate_model.py` → `04_fuse_and_evaluate.py`

Output streams into a `LogView` below the summary. Stops on first non-zero exit.

### 6.2 Synthetic Data
Split into three sections: Config form (top), action buttons (middle), log (bottom).

**Config form** edits `workspaces/<domain>/config.yaml`:
- `teacher.base_url`, `teacher.model`, `teacher.api_key` (masked)
- `generate.target_size`, `filter.judge.score_cutoff`
- `[Save]` button deep-merges edits into the domain config file.

**Action buttons:**
- `[Init]` → `python cli.py init <domain> --desc "<desc>"` (triggers a modal for desc/seeds path)
- `[Curate]` → opens `seeds/candidates.jsonl` in `$EDITOR` (suspends TUI), then runs `python cli.py curate <domain>` on resume
- `[Generate]` → `python cli.py generate <domain>`
- `[Prepare]` → `python cli.py prepare <domain> --system-prompt "<prompt>"` (triggers a modal for system prompt if not yet set)

Each button is disabled when its prerequisite state hasn't been reached (e.g. Generate is disabled until state ≥ `seeded`).

### 6.3 Training & Logs
**Config form** edits `config/model_config.yaml` and `config/training_config.yaml`:
- `base_model.path`, `lora.num_layers`, `training.learning_rate`, `training.iters`
- `[Save]` writes directly to the respective YAML files.

**Action:** `[▶ Train]` → `python scripts/02_train_model.py --model-config workspaces/<domain>/runtime_model_config.yaml --training-config workspaces/<domain>/runtime_training_config.yaml --train-data workspaces/<domain>/processed/train.json --val-data workspaces/<domain>/processed/val.json`

After the subprocess exits successfully, a loss sparkline is rendered from `workspaces/<domain>/logs/training/training_metrics.json` (train loss and val loss over iterations). The log remains visible above the sparkline.

### 6.4 Evaluation Report
**Actions:**
- `[▶ Evaluate]` → `python scripts/03_evaluate_model.py --config workspaces/<domain>/runtime_eval_config.yaml --adapters-path workspaces/<domain>/adapters --test-data workspaces/<domain>/processed/test.json`
- `[▶ Fuse & Evaluate]` → `python scripts/04_fuse_and_evaluate.py --model-config workspaces/<domain>/runtime_model_config.yaml --eval-config workspaces/<domain>/runtime_eval_config.yaml --test-data workspaces/<domain>/processed/test.json --adapters-path workspaces/<domain>/adapters --output-path workspaces/<domain>/fused`

After a run completes, a results table is rendered from `workspaces/<domain>/logs/evaluation/*.json`:

```
Model            BERTScore F1   Word Overlap
base_model       0.6821         0.3104
lora_runtime     0.7934         0.4217
fused            0.7931         0.4209
```

Table refreshes automatically on subprocess exit. Log streams below.

### 6.5 Deployment & Adapters
Displays the adapter inventory inferred from the filesystem.

```
workspaces/<domain>/adapters/    47 MB    [● active]
workspaces/<domain>/fused/       ---      [○ not fused]
```

**Actions:**
- `[▶ Fuse & Evaluate]` → `python scripts/04_fuse_and_evaluate.py --model-config workspaces/<domain>/runtime_model_config.yaml …` (same flags as Evaluation panel)
- `[Create Ollama Model]` → generates a `Modelfile` pointing at `workspaces/<domain>/fused/` and runs `ollama create <domain>-lora -f Modelfile`. Disabled until `workspaces/<domain>/fused/` is non-empty.

Output streams into `LogView` below.

---

## 7. Config Form Widget

`config_form.py` is a reusable Textual widget used by Synthetic and Training panels.

- Accepts a list of `(label, yaml_path, key_path)` tuples at construction.
- On mount: reads current values by deep-navigating `key_path` in the loaded YAML.
- Renders one `Input` per knob; password masking for `api_key` fields.
- `[Save]` deep-merges the edited values back into the YAML file (same `_deep_merge` from `config.py`). Does not restart any running subprocess.

---

## 8. New Domain Flow

`[+ New Domain]` opens a modal dialog:
1. Domain name (text input, validated: lowercase, no spaces)
2. Source: radio — "Bootstrap from description" or "Import from file"
3. If bootstrap: description textarea + optional system prompt
4. If import: file path input
5. `[Create]` → runs `python cli.py init <domain> [--desc | --seeds]` as a subprocess; on success the sidebar re-scans and selects the new domain.

---

## 9. Error Handling

- **Subprocess failure**: exit code ≠ 0 → action button turns red, error shown in log, status bar shows `✗ <command> failed (exit 1)`. User can re-run after fixing.
- **Missing prerequisite**: buttons gated by domain state; disabled state shows tooltip explaining what's needed.
- **Config save failure**: YAML write error shown inline in the config form (not a modal).
- **Ollama not installed**: `[Create Ollama Model]` checks `which ollama` before enabling; shows "ollama not found" tooltip if absent.

---

## 10. Technology

- **Framework:** Textual (`textual>=0.60`) + Rich
- **Config I/O:** reuses `config.py`'s `_deep_merge` and `load_config`
- **No new runtime dependencies** beyond Textual (already a natural fit given the stack)

---

## 11. Testing Strategy

- **Unit tests** for `domain.py` (state inference from mock filesystem), `config_form.py` (save round-trip), `runner.py` (message posting from a fake subprocess).
- **Textual `Pilot` tests** for key interactions: domain switching, button disabled states, config save, New Domain modal.
- **No integration test against real subprocesses** in CI — the runner is tested with a fake `echo`-based command.

---

## 12. Out of Scope (Future Slices)
- Multi-adapter runtime switching in Ollama
- Training progress beyond loss sparkline (e.g. live perplexity)
- Cloud training offload
- Web UI
- HuggingFace model upload (`scripts/upload_model_to_hf.py`) — button placeholder only
