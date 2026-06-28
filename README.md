# ElixirLoRA

Fine-tuning an LLM involves many moving parts — data preparation, training runs, evaluation, adapter fusion, and deployment. ElixirLoRA is a LoRA fine-tuning workbench for Apple Silicon that organizes the full pipeline into a single managed workspace per domain, driven by a guided TUI or CLI.

Each domain gets its own workspace under `workspaces/<domain>/`, tracking everything from raw seed examples through trained adapters to a deployable GGUF — so the state of any fine-tuning project is always visible and reproducible. Synthetic training data generation from a local teacher LLM is available as an optional step when you don't have enough hand-curated examples.

## Requirements

- Apple Silicon Mac (MLX)
- Python 3.11+
- A local OpenAI-compatible LLM server (e.g. [Ollama](https://ollama.com), llama.cpp) — only required for synthetic data generation

## Setup

```bash
CONDA_SUBDIR=osx-arm64 conda create -n elixirlora python=3.11
conda activate elixirlora
conda config --env --set subdir osx-arm64
pip install -r requirements.txt
```

## Quick Start

Launch the TUI to be guided through the full pipeline:

```bash
python3 cli.py tui
python3 cli.py tui --domain mymodel   # pre-select a domain
```

Or drive the pipeline directly from the CLI:

```bash
# 1. Create a domain workspace
python3 cli.py init mymodel --desc "A helpful coding assistant"

# 2. Add examples to workspaces/mymodel/seeds/candidates.jsonl, then curate
python3 cli.py curate mymodel

# 3. (Optional) Generate synthetic training data from a teacher LLM
python3 cli.py generate mymodel

# 4. Format into train/val/test splits
python3 cli.py prepare mymodel --system-prompt "You are a helpful coding assistant."

# 5. Fine-tune
python3 cli.py train mymodel --method sft \
  --model-config workspaces/mymodel/runtime_model_config.yaml \
  --training-config workspaces/mymodel/runtime_training_config.yaml \
  --train-data workspaces/mymodel/processed/train.json

# 6. Chat with your model
python3 cli.py chat mymodel
```

## Project Layout

```
cli.py                        # single entry point
commands/                     # one file per CLI subcommand
config/
  defaults.yaml               # global defaults (merged with per-domain config)
src/
  data/
    synthetic/                # synthetic data pipeline (pipeline.py is the entry)
  training/                   # sft.py (mlx-tune), dpo.py stub, grpo.py stub
  evaluation/                 # ModelEvaluator (word-overlap + BERTScore)
  inference/                  # ChatInterface, generator
  utils/                      # AdapterFusion, model_utils
tui/                          # Textual TUI (app.py + panels/)
workspaces/
  <domain>/                   # all per-domain artifacts
    seeds/candidates.jsonl    # raw seed examples
    seeds/approved.jsonl      # curated seeds (input to generate)
    generated/filtered.jsonl  # output of generate
    processed/                # train.json, val.json, test.json
    adapters/                 # trained LoRA weights
    fused/                    # merged model (ready to deploy)
    logs/training/
    config.yaml               # optional per-domain config overrides
```

## CLI Reference

All commands take `<domain>` as their first positional argument.

### `init`
Create a new domain workspace and seed file.

```bash
python3 cli.py init <domain> [--desc "Description"] [--seeds path/to/seeds.jsonl]
```

### `curate`
Promote `candidates.jsonl` to `approved.jsonl` (edit the candidates file first to remove low-quality seeds).

```bash
python3 cli.py curate <domain>
```

### `generate`
Call the teacher LLM to generate synthetic Q&A pairs from approved seeds, then filter by schema, deduplication, judge scoring, and diversity quotas.

```bash
python3 cli.py generate <domain>
```

Configuration lives in `config/defaults.yaml` (override in `workspaces/<domain>/config.yaml`):

```yaml
teacher:
  base_url: http://localhost:8080/v1
  model: qwen3.6
generate:
  target_size: 2000
filter:
  dedup: {embedding_model: all-MiniLM-L6-v2, similarity_threshold: 0.92}
  judge: {model: qwen3.6, score_cutoff: 4}
```

### `prepare`
Format `filtered.jsonl` into conversation-style JSON splits for training.

```bash
python3 cli.py prepare <domain> \
  --system-prompt "You are a helpful assistant." \
  [--test-split 0.1] [--val-split 0.1]
```

Output: `workspaces/<domain>/processed/{train,val,test}.json`

### `train`
Fine-tune using [mlx-tune](https://github.com/ARahim3/mlx-tune). SFT is production-ready; DPO and GRPO are stubs for future use.

```bash
python3 cli.py train <domain> \
  --method sft \
  --model-config  <path/to/model_config.yaml> \
  --training-config <path/to/training_config.yaml> \
  --train-data workspaces/<domain>/processed/train.json \
  [--val-data  workspaces/<domain>/processed/val.json]
```

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
```

Output: `workspaces/<domain>/adapters/`

### `evaluate`
Evaluate the base model and/or fine-tuned adapters.

```bash
python3 cli.py evaluate <domain> \
  --eval-config config/evaluation_config.yaml \
  [--adapters-path workspaces/<domain>/adapters] \
  [--model-config  workspaces/<domain>/runtime_model_config.yaml]
```

If `--model-config` is omitted the command looks for `runtime_model_config.yaml` in the workspace (written by the TUI). Pass it explicitly for CLI-only workflows.

### `fuse`
Merge LoRA adapter weights into the base model for single-file deployment.

```bash
python3 cli.py fuse <domain> \
  --model-config workspaces/<domain>/runtime_model_config.yaml \
  [--output-path workspaces/<domain>/fused] \
  [--eval-config config/evaluation_config.yaml]
```

### `chat`
Start an interactive chat session with the fused model (falls back to adapter runtime).

```bash
python3 cli.py chat <domain> [--fused/--no-fused] [--max-tokens 200] [--temperature 0.7]
```

Override the system prompt by adding a `chat.system_prompt` key to `workspaces/<domain>/config.yaml`.

### `upload`
Push the fused model to HuggingFace Hub.

```bash
HF_TOKEN=hf_... python3 cli.py upload <domain> \
  --repo-name username/my-model \
  [--private]
```

### `tui`
The primary interface. Guides you through the full pipeline — workspace management, training, evaluation, and deployment — with live log streaming and interactive config forms.

```bash
python3 cli.py tui [--domain <domain>]
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
  → bootstrap (LLM generates initial Q&A)
  → generate  (expand to target_size via few-shot prompting)
  → filter    (schema validation → dedup → judge scoring → diversity)
  → refine    (optional self-critique passes)
  → assemble  → workspaces/<domain>/generated/filtered.jsonl
```

The teacher LLM is called via the OpenAI-compatible API. Any server that exposes `/v1/chat/completions` works (Ollama, llama.cpp, vLLM).

## Training Methods

| Method | Status | Notes |
|--------|--------|-------|
| `sft`  | Ready  | Supervised fine-tuning via mlx-tune |
| `dpo`  | Stub   | Requires preference pairs |
| `grpo` | Stub   | Requires reward model |

## License

MIT — see [LICENSE](LICENSE).
