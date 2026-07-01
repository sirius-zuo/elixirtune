# Code Review Domain

Fine-tune Qwen2.5-7B to act as an automated code review assistant. This example downloads the
[ronantakizawa/github-codereview](https://huggingface.co/datasets/ronantakizawa/github-codereview)
dataset, converts it to ElixirTune seed format, and walks you through the full pipeline —
synthetic data generation, fine-tuning, fusion, and GGUF export.

```
seed data (GitHub PR reviews)
  → synthetic expansion (teacher model generates variations)
  → filter & deduplicate
  → fine-tune Qwen2.5-7B with LoRA
  → fuse & export to GGUF (for llama.cpp / Ollama)
```

## What You'll Get

After running the pipeline, `workspaces/code-review/fused/` contains a LoRA-fused model
that can review code diffs and produce structured feedback:

```
[review_type: style]
Found unused import: `os` in line 3. Consider removing it.

[suggestion]
```python
# Remove unused import
# import os
from pathlib import Path
```
```

## Prerequisites

- Apple Silicon Mac (MLX)
- Python 3.11+
- **Teacher model** running on `localhost:90909` (Qwen3.6-35B-A3B via llama.cpp)
  — this model generates synthetic training data. See `config/defaults.yaml`.
- Hugging Face account + token (for downloading the dataset and base model)
- `llama.cpp` (optional, only if you want GGUF export)

### Virtual Environment

All dependencies are already listed in `requirements.txt` and included in the
project `.venv`. Activate it before running any commands:

```bash
source .venv/bin/activate
```

## Quick Start

The simplest path is via the guided TUI:

```bash
# 1. Download dataset and create workspace
python3 examples/code_review/setup.py --domain code-review

# 2. Launch TUI (pre-selects the code-review domain)
python3 cli.py tui --domain code-review

# 3. Follow the TUI panels:
#    Synth  → generate synthetic training data
#    Train  → fine-tune with LoRA
#    Deploy → fuse adapters into base model
#    Export → convert to GGUF format
```

## Full CLI Walkthrough

You can also run each step manually:

```bash
# 1. Download dataset → workspaces/code-review/seeds/approved.jsonl
python3 examples/code_review/setup.py --domain code-review

# 2. (Optional) Edit seeds for quality control
#    workspaces/code-review/seeds/candidates.jsonl

# 3. Generate synthetic training data (calls teacher LLM)
python3 cli.py generate code-review

# 4. Format into train/val/test splits
python3 cli.py prepare code-review \
  --system-prompt "You are an expert code reviewer. Analyze the given code diff and provide structured feedback."

# 5. Fine-tune (downloads Qwen2.5-7B from HuggingFace automatically)
python3 cli.py train code-review \
  --method sft \
  --model-config workspaces/code-review/runtime_model_config.yaml \
  --training-config workspaces/code-review/runtime_training_config.yaml \
  --train-data workspaces/code-review/processed/train.json

# 6. Fuse LoRA weights into base model
python3 cli.py fuse code-review

# 7. Chat with the fine-tuned model
python3 cli.py chat code-review

# 8. Export to GGUF (requires llama.cpp)
python3 examples/code_review/export_gguf.py --domain code-review
```

## Setup Script Options

```bash
python3 examples/code_review/setup.py --domain code-review \
  --languages Python,TypeScript \
  --target 50000 \
  --no-negative
```

| Flag | Default | Description |
|------|---------|-------------|
| `--domain` | `code-review` | Workspace name under `workspaces/` |
| `--languages` | *(all)* | Comma-separated languages, e.g. `Python,TypeScript,JavaScript` |
| `--target` | `200000` | Target number of seed records (capped by dataset size) |
| `--no-negative` | off | Exclude "no issues found" examples |
| `--root` | `.` | Project root directory |

## Architecture

```
examples/code_review/
├── setup.py            # Dataset download & conversion (entry point)
├── export_gguf.py      # Standalone GGUF export wrapper
└── README.md           # This file
```

- **`setup.py`** downloads `ronantakizawa/github-codereview` from HuggingFace,
  filters by language/negative examples, converts each record into a conversation-style
  seed pair (`user`: code diff + context → `assistant`: structured review), and writes
  `workspaces/<domain>/seeds/approved.jsonl` + `config.yaml`.
- **`export_gguf.py`** is a convenience wrapper that calls
  `cli.py export-gguf <domain>` with the same quantization options.

## Configuration

Per-domain config is written to `workspaces/code-review/config.yaml`:

```yaml
generate:
  target_size: 200000
  fewshot_k: 4
filter:
  dedup:
    embedding_model: all-MiniLM-L6-v2
    similarity_threshold: 0.92
  diversity:
    quotas: {}
```

Override defaults by editing this file or `config/defaults.yaml`.

## Model Config

The fine-tuning base model is Qwen2.5-7B (4-bit MLX-quantized):

```yaml
# config/model_config.yaml (project-wide default)
base_model:
  path: mlx-community/Qwen2.5-7B-Instruct-4bit

lora:
  num_layers: 28
  lora_layers: 28
  rank: 16
  scale: 20.0
  dropout: 0.1
  keys:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
```

## Outputs

| Path | Description |
|------|-------------|
| `workspaces/code-review/seeds/approved.jsonl` | Curated seed review pairs |
| `workspaces/code-review/generated/filtered.jsonl` | Synthetic training data |
| `workspaces/code-review/processed/{train,val,test}.json` | Train splits |
| `workspaces/code-review/adapters/` | Trained LoRA weights |
| `workspaces/code-review/fused/` | Fused model (ready to chat/export) |
| `workspaces/code-review/fused/code-review.gguf` | GGUF export (after `export-gguf`) |

## GGUF Export

Export for use with **llama.cpp**, **Ollama**, or any GGUF-compatible runner:

```bash
# Default: Q4_K_M quantization
python3 examples/code_review/export_gguf.py --domain code-review

# Custom quantization
python3 examples/code_review/export_gguf.py --domain code-review --quantization Q8_0
```

Quantization options: `Q4_K_M` (default, ~4.5 GB), `Q5_K_M` (~5.3 GB), `Q8_0` (~8 GB).

### llama.cpp Installation

Required for GGUF export:

```bash
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
```

Ensure `llama-convert-hf-to-gguf.py` and `llama-quantize` are on your `PATH`
(they're installed to `build/bin/`).
