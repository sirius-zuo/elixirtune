# ElixirTune

Fine-tuning a model involves many moving parts — data preparation, training runs, evaluation, adapter fusion, and deployment. ElixirTune is a fine-tuning workbench for Apple Silicon that organizes the full pipeline into a single managed workspace per domain, driven by a guided TUI or CLI.

Each domain gets its own workspace under `workspaces/<domain>/`, tracking everything from raw seed examples through trained adapters to a deployable GGUF — so the state of any fine-tuning project is always visible and reproducible. Synthetic training data generation from a local teacher LLM is available as an optional step when you don't have enough hand-curated examples.

## Requirements

- Apple Silicon Mac (MLX)
- Python 3.11+
- A local OpenAI-compatible LLM server (e.g. [Ollama](https://ollama.com), llama.cpp) — required for synthetic data generation and DPO preference-data generation (used as the teacher / judge)

## Setup

The project runs in a virtualenv (`.venv`) where all dependencies — `mlx`, `mlx-tune`, `mlx-lm`, etc. — are installed:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Important — which Python:** run every `python cli.py …` command below **inside the activated environment**. A bare system `python3` will not have the dependencies (you'll hit `ModuleNotFoundError: No module named 'mlx_tune'`). Either keep `.venv` activated, or call it explicitly as `.venv/bin/python cli.py …`. The same applies to running the tests: `.venv/bin/python -m pytest`.

## Quick Start

Launch the TUI to be guided through the full pipeline:

```bash
python cli.py tui
python cli.py tui --domain mymodel   # pre-select a domain
```

Or drive the pipeline directly from the CLI:

```bash
# 1. Create a domain workspace
python cli.py init mymodel --desc "A helpful coding assistant"

# 2. Add examples to workspaces/mymodel/seeds/candidates.jsonl, then curate
python cli.py curate mymodel

# 3. (Optional) Generate synthetic training data from a teacher LLM
python cli.py generate mymodel

# 4. Format into train/val/test splits
python cli.py prepare mymodel --system-prompt "You are a helpful coding assistant."

# 5. Fine-tune
python cli.py train mymodel --method sft \
  --model-config workspaces/mymodel/runtime_model_config.yaml \
  --training-config workspaces/mymodel/runtime_training_config.yaml \
  --train-data workspaces/mymodel/processed/train.json

# 6. Chat with your model
python cli.py chat mymodel
```

> To further refine with preference optimization after SFT, see [Preference Optimization (DPO)](#preference-optimization-dpo).

## Project Layout

```
cli.py                        # single entry point
commands/                     # one file per CLI subcommand
config/
  defaults.yaml               # global defaults (merged with per-domain config)
src/
  data/
    synthetic/                # synthetic data pipeline (pipeline.py is the entry)
    dpo/                      # DPO preference-data pipeline (pipeline.py)
  training/                   # sft.py + dpo.py (mlx-tune), grpo.py stub
  evaluation/                 # ModelEvaluator (word-overlap + BERTScore)
  inference/                  # ChatInterface, generator
  utils/                      # AdapterFusion, model_utils
tui/                          # Textual TUI (app.py + panels/)
workspaces/
  <domain>/                   # all per-domain artifacts
    seeds/candidates.jsonl    # raw seed examples
    seeds/approved.jsonl      # curated seeds (input to generate)
    generated/filtered.jsonl  # output of generate
    processed/                # train.json, val.json, test.json (SFT)
    processed/dpo.json        # preference pairs (DPO) — from prepare-dpo
    adapters/                 # trained LoRA weights (SFT or DPO)
    fused/                    # merged model (ready to deploy)
    logs/training/
    config.yaml               # optional per-domain config overrides
```

## CLI Reference

All commands take `<domain>` as their first positional argument.

### `init`
Create a new domain workspace and seed file.

```bash
python cli.py init <domain> [--desc "Description"] [--seeds path/to/seeds.jsonl]
```

### `curate`
Promote `candidates.jsonl` to `approved.jsonl` (edit the candidates file first to remove low-quality seeds).

```bash
python cli.py curate <domain>
```

### `generate`
Call the teacher LLM to generate synthetic Q&A pairs from approved seeds, then filter by schema, deduplication, judge scoring, and diversity quotas.

```bash
python cli.py generate <domain>
```

Configuration lives in `config/defaults.yaml` (override in `workspaces/<domain>/config.yaml`):

```yaml
teacher:
  base_url: http://localhost:8080/v1
  model: qwen3.6
generate:
  target_size: 2000
  batch_size: 5        # distinct pairs requested per teacher call
  num_topics: 40       # diverse sub-topics planned per run to steer generation
filter:
  dedup: {embedding_model: all-MiniLM-L6-v2, similarity_threshold: 0.92}
  judge: {model: qwen3.6, score_cutoff: 4}
```

Add `--verbose` for full per-item request/response logging.

### `prepare`
Format curated seeds **and** generated data into conversation-style JSON splits for SFT training. Approved seeds are always included; generated `filtered.jsonl` is added when present, with exact-duplicate conversations removed.

```bash
python cli.py prepare <domain> \
  [--system-prompt "You are a helpful assistant."] \
  [--test-split 0.1] [--val-split 0.1]
```

`--system-prompt` is optional — if omitted it resolves from `prepare.system_prompt` in the domain config, falling back to a generic default.

Output: `workspaces/<domain>/processed/{train,val,test}.json`

### `train`
Fine-tune using [mlx-tune](https://github.com/ARahim3/mlx-tune). `sft` and `dpo` are supported; `grpo` is a stub.

```bash
# SFT (supervised fine-tuning)
python cli.py train <domain> \
  --method sft \
  --model-config  <path/to/model_config.yaml> \
  --training-config <path/to/training_config.yaml> \
  --train-data workspaces/<domain>/processed/train.json \
  [--val-data  workspaces/<domain>/processed/val.json]

# DPO (preference optimization) — train on preference pairs (see prepare-dpo)
python cli.py train <domain> \
  --method dpo \
  --model-config  <path/to/model_config.yaml> \
  --training-config <path/to/training_config.yaml> \
  --train-data workspaces/<domain>/processed/dpo.json
```

By default DPO **continues from the SFT-fused model** (`workspaces/<domain>/fused`), so run SFT → `fuse` before DPO. Set `dpo.from_base: true` in the training config to instead train a fresh LoRA on the base model. DPO has no eval split.

`model_config.yaml` shape:

```yaml
base_model:
  path: mlx-community/Phi-3-mini-4k-instruct-4bit
lora:
  rank: 16
  scale: 20.0
  dropout: 0.1
  keys: [q_proj, v_proj]
```

`training_config.yaml` shape:

```yaml
training:
  iters: 2000
  batch_size: 4
  learning_rate: 1e-5
  steps_per_eval: 50
dpo:                  # only used when --method dpo
  beta: 0.1           # preference strength
  from_base: false    # false = continue from SFT-fused model; true = fresh LoRA on base
```

Output: `workspaces/<domain>/adapters/`

### `prepare-dpo`
Build DPO preference pairs (`processed/dpo.json`). For each prompt it gathers candidate answers from configurable sources — **teacher samples**, the **SFT-fused model**, and the **base model** — judge-scores each, and pairs the best as `chosen` and the worst as `rejected` when the score gap is wide enough.

```bash
python cli.py prepare-dpo <domain> \
  [--model-config workspaces/<domain>/runtime_model_config.yaml] \
  [--teacher-samples 2] [--use-sft/--no-use-sft] [--use-base/--no-use-base] \
  [--min-margin 2] [--max-prompts 200] [--max-tokens 256]
```

Defaults live under `dpo_data` in `config/defaults.yaml` (override per-domain in `config.yaml`):

```yaml
dpo_data:
  teacher_samples: 2   # teacher candidate answers per prompt (0 to disable)
  use_sft: true        # include the SFT-fused model as a candidate source
  use_base: true       # include the base model as a candidate source
  min_margin: 2        # min judge-score gap (1-5) to keep a pair
  max_prompts: 200
  max_tokens: 256
```

Output: `workspaces/<domain>/processed/dpo.json`

### `evaluate`
Evaluate the base model and/or fine-tuned adapters.

```bash
python cli.py evaluate <domain> \
  --eval-config config/evaluation_config.yaml \
  [--adapters-path workspaces/<domain>/adapters] \
  [--model-config  workspaces/<domain>/runtime_model_config.yaml]
```

If `--model-config` is omitted the command looks for `runtime_model_config.yaml` in the workspace (written by the TUI). Pass it explicitly for CLI-only workflows.

### `fuse`
Merge LoRA adapter weights into the base model for single-file deployment.

```bash
python cli.py fuse <domain> \
  --model-config workspaces/<domain>/runtime_model_config.yaml \
  [--output-path workspaces/<domain>/fused] \
  [--eval-config config/evaluation_config.yaml]
```

### `chat`
Start an interactive chat session with the fused model (falls back to adapter runtime).

```bash
python cli.py chat <domain> [--fused/--no-fused] [--max-tokens 200] [--temperature 0.7]
```

Override the system prompt by adding a `chat.system_prompt` key to `workspaces/<domain>/config.yaml`.

### `upload`
Push the fused model to HuggingFace Hub.

```bash
HF_TOKEN=hf_... python cli.py upload <domain> \
  --repo-name username/my-model \
  [--private]
```

### `tui`
The primary interface. Guides you through the full pipeline — workspace management, training, evaluation, and deployment — with live log streaming and interactive config forms.

```bash
python cli.py tui [--domain <domain>]
```

## Configuration

Global defaults are in `config/defaults.yaml`. Per-domain overrides go in `workspaces/<domain>/config.yaml` — keys are deep-merged so you only need to specify what differs.

```yaml
# workspaces/mymodel/config.yaml
generate:
  target_size: 500          # smaller run for testing
teacher:
  model: llama3.2           # use a different model for this domain
chat:
  system_prompt: "You are an expert in Elixir."
```

## Synthetic Data Generation (Optional)

When you don't have enough hand-curated training examples, the `generate` command can synthesize more from a local teacher LLM. It runs this pipeline internally:

```
approved seeds
  → plan topics  (teacher lists diverse sub-topics from the domain + seeds)
  → generate     (batch-generate distinct pairs, each steered to a different topic)
  → refine       (optional self-critique passes)
  → filter       (schema validation → dedup → judge scoring → diversity)
  → assemble     → workspaces/<domain>/generated/filtered.jsonl
```

Topic steering + batch generation (`batch_size`, `num_topics`) keep the output diverse instead of mode-collapsing into near-identical answers. Pass `--verbose` to see the full request/response per item and per-item judge scores in the log.

The teacher LLM is called via the OpenAI-compatible API. Any server that exposes `/v1/chat/completions` works (Ollama, llama.cpp, vLLM).

## Training Methods

| Method | Status | Notes |
|--------|--------|-------|
| `sft`  | Ready  | Supervised fine-tuning via mlx-tune |
| `dpo`  | Ready  | Preference optimization; continues from the SFT-fused model by default |
| `grpo` | Stub   | Requires reward model |

## Preference Optimization (DPO)

DPO refines a model from **preference pairs** (`{prompt, chosen, rejected}`) rather than single target answers. It builds on SFT — the typical flow is:

```
curate / generate  →  SFT (train --method sft)  →  fuse  →  prepare-dpo  →  DPO (train --method dpo)
```

1. **SFT then fuse** — produce `workspaces/<domain>/fused/`. DPO continues from these fine-tuned weights by default (mlx-tune can't resume a raw adapter, so DPO starts from the fused model; the frozen DPO reference is built internally).
2. **`prepare-dpo`** — for each prompt, generate candidate answers from the configured sources (teacher × K, the SFT-fused model, the base model), judge-score them, and keep best/worst pairs whose score gap ≥ `min_margin` → `processed/dpo.json`.
3. **`train --method dpo`** — optimize on those pairs (controlled by `dpo.beta`), writing the DPO adapter to `adapters/`.

In the TUI this is the **Training** tab: pick **DPO** in the method selector, click **Prepare DPO data**, then **Train**.

## License

MIT — see [LICENSE](LICENSE).
