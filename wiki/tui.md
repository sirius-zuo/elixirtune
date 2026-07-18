# TUI (Terminal User Interface)

## Purpose

The TUI provides an interactive, domain-centric interface for managing ElixirTune's full pipeline — from domain creation through data generation, training, evaluation, and deployment. Built on **Textual**, it presents a sidebar for domain navigation, a tabbed content area for domain-specific panels, and a shared subprocess runner for executing CLI commands with live log streaming. The TUI dynamically adapts its panels based on the selected domain's type (LM vs. embedding).

## Position in the System

Consumed by:
- **[cli-commands](cli-commands.md)** — `cli.py tui` entry point launches `ElixirTuneApp`; TUI panels invoke CLI commands as subprocesses
- **[workspaces](workspaces.md)** — reads workspace directory structure to infer domain status; generates runtime configs

Consumes:
- **[data](data.md)** — synthetic pipeline, data preparation
- **[training](training.md)** — training execution and metrics polling
- **[evaluation](evaluation.md)** — evaluation runs and result display

## Architecture

```mermaid
classDiagram
    class ElixirTuneApp
    class Sidebar
    class BasePanel
    class SyntheticPanel
    class TrainingPanel
    class EvaluationPanel
    class DeploymentPanel
    class ChatPanel
    class EmbeddingTrainingPanel
    class EmbeddingEvalPanel
    class OverviewPanel
    class RunnerOutput
    class RunnerDone
    class LogView
    class ConfigForm

    ElixirTuneApp --> Sidebar
    ElixirTuneApp --> BasePanel
    ElixirTuneApp --> TabbedContent

    BasePanel <|-- SyntheticPanel
    BasePanel <|-- TrainingPanel
    BasePanel <|-- EvaluationPanel
    BasePanel <|-- DeploymentPanel
    BasePanel <|-- EmbeddingTrainingPanel
    BasePanel <|-- EmbeddingEvalPanel
    BasePanel <|-- OverviewPanel

    TrainingPanel --> LogView
    SyntheticPanel --> LogView
    TrainingPanel --> ConfigForm
    SyntheticPanel --> ConfigForm

    Sidebar --> DomainSelected
    Sidebar --> NewDomainRequested
    Sidebar --> DeleteDomainRequested

    class DomainSelected
    class NewDomainRequested
    class DeleteDomainRequested

    class RunnerOutput
    class RunnerDone

    RunnerOutput : line: str
    RunnerDone : exit_code: int, tag: str
```

**ElixirTuneApp** (`tui/app.py`): The main Textual App. Composes a Header, Sidebar, and TabbedContent with all panels. Domain switching (`_switch_domain`) updates all panels' `domain` reactive and shows/hides tabs based on domain type. The `_update_tabs_for_type` method uses pre-defined tab sets:
- **LM tabs:** Overview, Synth, Training, Eval, Deploy, Chat
- **Embedding tabs:** Overview, Embed Data, Embed Train, Embed Eval

**BasePanel** (`tui/app.py`): Abstract base for all domain panels. Has a reactive `domain` attribute — when changed, `refresh_content()` is called. All panels inherit from this to get domain-aware behavior.

**Sidebar** (`tui/sidebar.py`): Lists available domains with status badges. Dispatches `DomainSelected`, `NewDomainRequested`, and `DeleteDomainRequested` messages to the app.

**Runner** (`tui/runner.py`): Shared subprocess infrastructure. `stream_subprocess()` runs a command and yields `(line, None)` per output line, then `(None, exit_code)`. It handles raw bytes, strips ANSI escape sequences, and treats `\r` (carriage return) as a line reset — so progress bars show in-place updates. `RunnerOutput` and `RunnerDone` are Textual Messages for the event system.

**Widgets:**
- `ConfigForm` (`tui/widgets/config_form.py`): A form that reads config values from YAML files and writes changes back. Used by SyntheticPanel and TrainingPanel to edit pipeline parameters in-place.
- `LogView` (`tui/widgets/log_view.py`): A scrollable text display for subprocess output.
- `SectionRule` (`tui/widgets/section_rule.py`): A horizontal rule with a section title.

**Panels:**
- **SyntheticPanel** (`tui/panels/synthetic.py`): Domain lifecycle (init → curate → generate → prepare). Config form for teacher URL, model, API key, target size, batch size, topics, judge cutoff. Verbose log toggle.
- **TrainingPanel** (`tui/panels/training.py`): SFT and DPO training. Config form for base model, LoRA layers, learning rate, iterations. Method selector (SFT/DPO). Parses trainer stdout for loss/step via regex (`_DPO_STEP_RE`, `_SFT_TRAIN_RE`, `_SFT_VAL_RE`). Captures metrics in real-time and writes to `training_metrics.json` for TUI polling.
- **EvaluationPanel** (`tui/panels/evaluation.py`): Runs evaluation on the selected domain.
- **DeploymentPanel** (`tui/panels/deployment.py`): GGUF export and model deployment.
- **ChatPanel** (`tui/panels/chat.py`): Interactive chat with the trained model.
- **EmbeddingTrainingPanel** (`tui/panels/embedding_training.py`): Embedding data import/convert, bi-encoder training, cross-encoder training.
- **EmbeddingEvalPanel** (`tui/panels/embedding_eval.py`): Cosine similarity probe, Recall@K, BEIR benchmark, cross-encoder reranking.
- **OverviewPanel** (`tui/panels/overview.py`): Domain status and summary display.

## Runtime Flows

1. **Launch:** `cli.py tui [domain]` → `ElixirTuneApp(initial_domain=domain).run()`
   1. On mount, scan domains via `tui/domain.py:scan_domains()`
   2. Select the initial domain (CLI arg, first domain, or None)
   3. Set domain on all panels, update tab visibility based on domain type

2. **Domain switch:** Sidebar click → `DomainSelected` → `_switch_domain()` → update all panels' `domain` reactive → `_update_tabs_for_type()` → show/hide tabs
   1. Read domain type via `read_domain_type(ws)`
   2. Show LM tabs (`_LM_TABS`) or embedding tabs (`_EMBED_TABS`)
   3. Overview tab always visible

3. **Training execution:** TrainingPanel "▶ Train" → `_run_train()` → subprocess call to `cli.py train <domain> --method <sft|dpo>`
   1. Start with empty metrics: `{"train_loss": [], "val_loss": [], "iterations": []}`
   2. Stream subprocess stdout line by line
   3. `_capture_metric()` parses each line for loss/step via regex, updates metrics dict
   4. Writes metrics to `workspaces/{domain}/logs/training/training_metrics.json` (also written by `MetricsWriterCallback`)
   5. On `RunnerDone`, update UI, re-enable train button, trigger domain rescan

4. **Domain creation:** Sidebar "+" → `NewDomainScreen` → user enters name, description, model, and type (LM/Embedding) → creates workspace directory, writes `config.yaml`, `description.txt`, `runtime_config.yaml` files

5. **Domain deletion:** Sidebar delete → `DeleteDomainScreen` → confirms with user → deletes workspace directory → updates sidebar and panel state

## Key Decisions

### Tab visibility based on domain type
- **Decision:** The TUI shows different tabs depending on whether the selected domain is an `lm` or `embedding` type. The Overview tab is always visible.
- **Context:** LM and embedding domains have fundamentally different pipelines — different panels, different data formats, different training methods. Showing all panels for all domains would be confusing.
- **Alternatives rejected:** Single panel with conditional content (adds complexity to every panel); separate apps per domain type (loses domain management convenience).
- **Consequences:** The tab sets are defined as `_LM_TABS` and `_EMBED_TABS` constants in `app.py`. The `_update_tabs_for_type()` method shows/hides tabs by ID.
- **Ref:** 2026-06-30, Embedding Rename Design Spec §2

### Regex-based metric parsing from trainer stdout
- **Decision:** The TrainingPanel parses trainer progress lines via regex patterns (`_DPO_STEP_RE` for DPO, `_SFT_TRAIN_RE` and `_SFT_VAL_RE` for SFT) to extract loss and step values in real-time.
- **Context:** `mlx_tune` trainers stream progress to stdout in a text format ("Step 5/100 | Loss: 1.2345" for DPO, "Iter 5: Train loss 1.234, ..." for SFT). The TUI needs to parse this to display real-time metrics.
- **Alternatives rejected:** Parsing the `training_metrics.json` file (already written by `MetricsWriterCallback`, but the TUI also parses stdout to fill gaps for older runs and to display the progress label before the file is written).
- **Consequences:** The regex patterns must match the trainer's stdout format. If the format changes, the regexes need updating. This is fragile but functional.
- **Ref:** 2026-06-26, Training Backend Refactor Design Spec

### Subprocess runner with ANSI stripping and carriage return handling
- **Decision:** `stream_subprocess()` handles raw subprocess output by treating `\r` (carriage return) as a line reset, stripping ANSI escape sequences, and yielding structured messages.
- **Context:** Trainer progress bars and in-place updates use `\r` to overwrite the current line. A naive line-based reader would accumulate all intermediate states (e.g., "Step 1/100", "Step 2/100", ...) instead of showing only the final state.
- **Alternatives rejected:** Using `subprocess.Popen(stream=True)` with `tee` (would show all intermediate states); capturing full output at the end (no live feedback).
- **Consequences:** The TUI's log view shows only the final state of in-place updates, matching what a terminal would display. Progress bars and loss updates appear as single lines.
- **Ref:** 2026-06-26, Training Backend Refactor Design Spec; commit 4624a64

### Runtime config generation per workspace
- **Decision:** The TUI generates workspace-specific `runtime_model_config.yaml`, `runtime_training_config.yaml`, and `runtime_eval_config.yaml` files by overlaying workspace paths on top of global defaults.
- **Context:** Global configs (`config/model_config.yaml`, etc.) don't know about workspace paths. Commands need workspace-scoped paths for adapters, data, logs, etc.
- **Alternatives rejected:** Hardcoding paths in commands (less flexible); passing paths as CLI arguments for everything (clutters the interface).
- **Consequences:** `tui/domain.py:generate_runtime_configs()` creates the overlay. Commands read from these runtime configs instead of the global defaults.
- **Ref:** 2026-06-26, Training Backend Refactor Design Spec

### ConfigForm for in-place parameter editing
- **Decision:** Panels use a `ConfigForm` widget that reads config values from YAML files and writes changes back, rather than using raw Textual Input widgets.
- **Context:** Pipeline parameters (teacher URL, LoRA rank, learning rate) are stored in YAML. Users need to edit them in the TUI without leaving the interface.
- **Alternatives rejected:** Custom widgets per panel (duplicates logic); external editor (breaks the flow).
- **Consequences:** `ConfigForm` is configured with `ConfigField` objects that specify the config file path, key path, and display label. It handles deep-merge writes back to the domain config.
- **Ref:** 2026-06-25, Synthetic Data Pipeline Design Spec §9

## Implementation Notes

- **Tab IDs are hardcoded:** The tab set constants (`_LM_TABS`, `_EMBED_TABS`) list tab IDs as strings. If a tab is renamed or its `id` attribute changes, the visibility logic must be updated.
- **TrainingPanel metric parsing regexes are fragile:** If the trainer's stdout format changes (e.g., mlx-tune updates its logging), the regexes must be updated. No fallback mechanism exists.
- **Embed Data tab is a placeholder:** The "Embed Data" tab (`tab-embed-data`) currently yields only a `Label` widget with instructions. The actual import/convert buttons are in the EmbeddingTrainingPanel. This is likely a design oversight or a work-in-progress.
- **No PR or design doc records a rationale for the Embed Data tab being a placeholder; observed current state: the tab exists but contains only instructional text. The actual embedding data operations are in the EmbeddingTrainingPanel.**
- **ChatPanel is dynamically imported:** In `app.py:compose()`, ChatPanel is imported inside the method (`from tui.panels.chat import ChatPanel`) to avoid a circular import with the other panels. This is a pattern used to break the import cycle.
- **ConfigForm reads from domain-specific paths:** SyntheticPanel's config form reads from `workspaces/{domain}/config.yaml`, not from the global `config/defaults.yaml`. This means each domain can have different teacher URLs, API keys, and pipeline parameters.

## Source Anchors

- `tui/app.py`
- `tui/domain.py`
- `tui/runner.py`
- `tui/sidebar.py`
- `tui/new_domain.py`
- `tui/delete_domain.py`
- `tui/gguf_export_modal.py`
- `tui/upload_modal.py`
- `tui/panels/synthetic.py`
- `tui/panels/training.py`
- `tui/panels/evaluation.py`
- `tui/panels/deployment.py`
- `tui/panels/embedding_training.py`
- `tui/panels/embedding_eval.py`
- `tui/panels/chat.py`
- `tui/panels/overview.py`
- `tui/widgets/config_form.py`
- `tui/widgets/log_view.py`
- `tui/widgets/section_rule.py`
- `docs/superpowers/specs/2026-06-25-synthetic-data-pipeline-design.md`
- `docs/superpowers/specs/2026-06-25-tui-dashboard-design.md`
- `docs/superpowers/specs/2026-06-26-third-slice-design.md`
- `docs/superpowers/specs/2026-06-30-elixirtune-embedding-rename-design.md`
- `docs/superpowers/specs/2026-07-01-deploy-gguf-export-config-modal-design.md`

## Related Pages

- [cli-commands](cli-commands.md)
- [data](data.md)
- [training](training.md)
- [evaluation](evaluation.md)
- [config](config.md)
- [workspaces](workspaces.md)
