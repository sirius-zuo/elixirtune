# Code Review Domain

Fine-tune a model to act as a code review assistant using the
[ronantakizawa/github-codereview](https://huggingface.co/datasets/ronantakizawa/github-codereview)
dataset.

## Quick Start

```bash
# 1. Download dataset and create workspace
python3 examples/code_review/setup.py --domain code-review

# 2. (Optional) Curate seeds by editing:
#    workspaces/code-review/seeds/candidates.jsonl

# 3. Launch the ElixirLoRA TUI
python3 cli.py tui --domain code-review

# 4. Follow the TUI: Synth → Training → Deploy → Export GGUF
```

## CLI Options

```bash
python3 examples/code_review/setup.py --domain myreview \
  --languages Python,TypeScript \
  --target 50000 \
  --no-negative
```

- `--domain`: Workspace name (default: `code-review`)
- `--languages`: Comma-separated language filter (default: all)
- `--target`: Target number of samples (default: 200000)
- `--no-negative`: Exclude "no issues found" examples (default: include)

## Output

Creates `workspaces/<domain>/seeds/approved.jsonl` with conversation-style
pairs ready for ElixirLoRA's generate/prepare/train pipeline.
