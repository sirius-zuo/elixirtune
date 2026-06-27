# Code Review Domain Design

**Status:** Approved
**Date:** 2026-06-27
**Scope:** Add a new code-review domain to ElixirLoRA using the `ronantakizawa/github-codereview` dataset, plus framework-level GGUF export support.

---

## Overview

ElixirLoRA will support a `code-review` domain that fine-tunes a model to act as a code review assistant. The model will:

1. **Classify review type** — suggestion, question, refactor, or none (clean code)
2. **Generate review content** — natural language comments with code suggestions

The output format is structured: `[review_type: <type>]` as a prefix, followed by natural language content. This is machine-parseable but still human-readable.

### Key Design Decisions

- **No framework changes for code-review data** — all dataset handling lives in `examples/code-review/`
- **GGUF export is framework-level** — belongs in ElixirLoRA core (`commands/export_gguf.py`), not in any example
- **TUI-first workflow** — dataset download integrated into the TUI's "New Domain" screen; full pipeline driven through the TUI

---

## Architecture

### Component Boundary

```
examples/code-review/       ← code-review-specific (dataset download + conversion)
ElixirLoRA core              ← framework-level (pipeline, TUI, GGUF export)
                              commands/export_gguf.py   ← GGUF conversion
                              tui/new_domain.py         ← new radio option
                              cli.py                    ← export-gguf subcommand
```

ElixirLoRA core remains untouched in its existing pipeline logic. The code-review example is a self-contained folder that prepares data, then delegates to ElixirLoRA for the rest of the workflow.

### Data Flow

```
ronantakizawa/github-codereview (HF, 218K rows)
  ↓
examples/code-review/setup.py  (download + format conversion)
  ↓
workspaces/<domain>/seeds/approved.jsonl  (ElixirLoRA seed format)
  ↓
ElixirLoRA TUI → Synth tab → (generate is optional: with 200K+ seeds from dataset, the existing `generate` step's bootstrap+expand pipeline is redundant — seeds go straight to `prepare`)
ElixirLoRA TUI → Training tab → train LoRA → adapters/
ElixirLoRA TUI → Deploy tab → fuse → fused/
ElixirLoRA TUI/CLI → Export GGUF → .gguf
```

---

## Data Format

### Dataset Schema

Each row from `ronantakizawa/github-codereview` contains:

| Field | Type | Description |
|---|---|---|
| `pr_title` | string | Pull request title |
| `repo_name` | string | Full repo name (e.g., "4ian/GDevelop") |
| `pr_number` | int | PR number |
| `code_diff` | string | Diff hunks (unified diff format) |
| `file_path` | string | Path to the file in the repo |
| `language` | string | Programming language |
| `review_type` | string | `suggestion`, `question`, `refactor`, or none |
| `suggestion` | string | The code suggestion (may be empty for "none") |
| `context` | string | Surrounding code context |
| `comment` | string | Reviewer's inline comment |
| `line` | int | Current line number in the diff |
| `old_line` | int | Old file line number |
| `new_line` | int | New file line number |

### ElixirLoRA Seed Format

Each dataset row is converted to one ElixirLoRA seed:

```json
{
  "instruction": "You are reviewing the following code:\n\n```diff\n@@ -269,3 +269,8 @@ namespace gdjs {\n- this.owner.setWidth(right - left);\n- this.owner.setX(left);\n+ const width = right - left;\n+ this...\n```\n\nFile: Extensions/AnchorBehavior/anchorruntimebehavior.ts\nLanguage: TypeScript\nPR: Fix anchor behavior when objects has custom origin\n\nPlease provide a code review.",
  "output": "[review_type: suggestion] Consider using `for...of` here since the index `i` isn't used.\n\n```suggestion\nfor (const item of items) { ... }\n```",
  "review_type": "suggestion",
  "language": "TypeScript",
  "repo": "4ian/GDevelop",
  "pr_title": "Add dispose method to Runtimegame"
}
```

Negative examples (51K rows where no issues were found) become:
```
"output": "[review_type: none] No issues found."
```

### Output Format (Fine-Tuned Model)

The fine-tuned model will produce output like:

```
[review_type: suggestion] Consider using a dedicated type annotation here for clarity.

```suggestion
const count: number = items.length;
```
```

The `[review_type: <type>]` prefix is machine-parseable. The rest is natural language that a human can read.

---

## ElixirLoRA Framework Changes

### 1. GGUF Export Command

**Location:** `commands/export_gguf.py` + `cli.py`

**CLI:**
```bash
python3 cli.py export-gguf <domain> [--quantization Q4_K_M] [--output output.gguf]
```

**Behavior:**
- Loads the fused model from `workspaces/<domain>/fused/`
- Converts to GGUF via mlx-llm → llama.cpp pipeline
- Supports quantization options: Q4_K_M (default), Q5_K_M, Q8_0
- Outputs to `workspaces/<domain>/fused/<domain>.gguf` or user-specified path

**Preconditions:**
- Model must be fused (or have a valid runtime_model_config.yaml)
- llama.cpp must be available (check on startup, offer install instructions if missing)

**TUI Integration:** New button "Export GGUF" in the Deploy tab, disabled until model is fused.

### 2. TUI: New Domain Radio Option

**Location:** `tui/new_domain.py`

**Changes:**
- Add third radio option: "Download code review dataset"
- When selected, runs `examples/code-review/setup.py --domain <name>` to download and prepare
- On success, switches to the new domain

---

## `examples/code-review/` Structure

```
examples/code-review/
├── setup.py              # Dataset download + conversion (main entry)
├── export_gguf.py        # Standalone GGUF export wrapper (optional, no TUI)
└── README.md             # Usage instructions
```

### `setup.py`

**CLI:**
```bash
python3 examples/code-review/setup.py --domain code-review
```

**Behavior:**
- Downloads `ronantakizawa/github-codereview` from HuggingFace
- Converts dataset rows to ElixirLoRA seed format
- Creates `workspaces/<domain>/seeds/candidates.jsonl` and `approved.jsonl`
- Creates `workspaces/<domain>/config.yaml` with code-review-specific config
- All data cached in `data/code-review/` to avoid re-downloading

**Options:**
- `--domain <name>` — workspace name (default: `code-review`)
- `--languages <lang1,lang2,...>` — filter to specific languages (default: all)
- `--target <N>` — target number of samples (default: 200000)
- `--no-negative` — exclude negative examples (default: include)

### `export_gguf.py` (Standalone)

For users who prefer CLI-only workflow (no TUI). Wraps `cli.py export-gguf`:
```bash
python3 examples/code-review/export_gguf.py --domain code-review
```

This is a thin wrapper (~10 lines) that calls `cli.py export-gguf` with appropriate args. Lower priority than setup.py but should be included for completeness.

### `README.md`

Usage instructions, explaining:
1. Run setup to download dataset
2. Launch TUI to run the pipeline
3. Export GGUF when done

---

## Training Configuration

- **Base model:** `mlx-community/Phi-3-mini-4k-instruct-4bit` (or similar instruct model, configurable)
- **LoRA rank:** 16 (configurable)
- **Target samples:** ~200K (167K positive + 51K negative)
- **Train/val/test split:** 80/10/10 (configurable)

---

## Error Handling

| Scenario | Behavior |
|---|---|
| HF download fails | Clear error with retry suggestion and network troubleshooting tips |
| GGUF export: llama.cpp missing | Pre-check on startup, offer install instructions |
| Insufficient disk space for GGUF export | Pre-check before starting, warn with expected size |
| Dataset conversion error | Report which rows failed, skip them, continue with valid rows |
| Training failure during code-review pipeline | TUI shows error in log view, keeps pipeline state for retry |

---

## Testing

### Unit Tests (`tests/test_code_review/`)
- `test_setup.py` — dataset download and format conversion logic
- `test_format_converter.py` — ElixirLoRA seed format generation

### Integration Test
- Run full pipeline on a tiny subset (10 samples) to verify:
  - Dataset downloads correctly
  - Seeds are valid ElixirLoRA format
  - Generate, prepare, train complete successfully
  - Model produces parseable output

---

## Files Changed

### ElixirLoRA Core (3 files)
1. `commands/export_gguf.py` — **new** — GGUF export command
2. `cli.py` — add `export-gguf` subcommand
3. `tui/new_domain.py` — add third radio option

### `examples/code-review/` (3 files)
1. `examples/code-review/setup.py` — **new** — dataset download + conversion
2. `examples/code-review/export_gguf.py` — **new** — standalone GGUF wrapper
3. `examples/code-review/README.md` — **new** — usage documentation

### Tests (2 files)
1. `tests/test_code_review/test_setup.py` — **new**
2. `tests/test_code_review/test_format_converter.py` — **new**

---

## Implementation Order

1. **`examples/code-review/setup.py`** — dataset download and conversion (independent, no framework changes)
2. **`examples/code-review/README.md`** — usage docs
3. **GGUF export** (`commands/export_gguf.py` + CLI + TUI button) — framework-level, independent
4. **TUI integration** — new radio option in NewDomainScreen
5. **`examples/code-review/export_gguf.py`** — standalone wrapper (~10 lines, lower priority)
6. **Tests**

Steps 1-2 and 3-4 are independent and can be worked on in parallel by separate agents.
