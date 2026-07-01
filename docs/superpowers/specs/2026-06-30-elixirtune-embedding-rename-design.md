# ElixirTune: Rename + Embedding Fine-Tuning Support

**Date:** 2026-06-30  
**Status:** Approved

---

## Overview

Two changes shipped together:

1. **Rename** `elixirlora` → `elixirtune` across all code, docs, and repository references. The internal Python module structure (`src.*`) is unaffected.
2. **Embedding fine-tuning support** via `mlx_tune`'s `FastEmbeddingModel` / `EmbeddingSFTTrainer` / `EmbeddingSFTConfig`, with a domain-type system that keeps the LM and embedding workflows cleanly separated.

---

## Section 1: Rename

**Scope of changes:**
- All Python source files: string literals, CLI help text, docstrings referencing "elixirlora"
- `README.md`, `docs/`, `config/` YAML comments
- TUI title in `tui/app.py`
- `workspaces/*/description.txt` files (where they reference the project name)
- `cli.py` help strings

**What stays the same:**
- Internal Python module paths — the package is imported as `src.*`, not `elixirlora.*`, so no import rewiring is needed.
- Workspace directory structure and all stored config files.

**Manual steps (post-commit, done by the user):**
- `mv /Users/jinzuo/projects/elixirlora /Users/jinzuo/projects/elixirtune`
- `gh repo rename elixirtune` (renames the GitHub remote)

---

## Section 2: Domain Type System

### `config.yaml` schema

Each domain's `config.yaml` gains a top-level `type` field:

```yaml
type: lm        # or: embedding
```

Existing domains without this field default to `lm` — fully backward compatible, no migration required.

### Domain creation (`tui/new_domain.py`)

The creation wizard gains a type selector step presented before the model/name fields:

- "Language Model" → writes `type: lm`
- "Embedding Model" → writes `type: embedding`

### `tui/domain.py`

New helper:

```python
def read_domain_type(ws: Path) -> str:
    """Returns 'lm' or 'embedding'. Defaults to 'lm' if absent."""
```

`infer_status` branches on domain type:

- `lm` status ladder: `EMPTY → SEEDED → PREPARED → TRAINED → FUSED → EXPORTED` (unchanged)
- `embedding` status ladder: `EMPTY → DATA_READY → PREPARED → TRAINED → CE_TRAINED`
  - `DATA_READY`: any raw data is present — seeds (`seeds/approved.jsonl`) for convert mode, or any file in `workspaces/<domain>/data/raw/` for import mode
  - `PREPARED`: `embedding_train.json` + `embedding_val.json` exist in `processed/`
  - `TRAINED`: bi-encoder adapters directory (`adapters/`) exists
  - `CE_TRAINED`: cross-encoder adapters directory (`ce_adapters/`) exists (optional stage)

### TUI panel routing (`tui/app.py`)

When a domain is selected, the app reads `read_domain_type(ws)` and mounts the appropriate panel set:

| Domain type | Panels |
|---|---|
| `lm` | Overview, Synthetic, Training, Evaluation, Deployment, Chat (unchanged) |
| `embedding` | Overview, Embedding Data, Embedding Training, Embedding Eval |

The sidebar domain list shows a small type badge: `[LM]` or `[EM]`.

---

## Section 3: Embedding Training Backend

### `src/training/embedding.py` (new)

Mirrors the structure of `sft.py`. Uses `FastEmbeddingModel`, `EmbeddingSFTTrainer`, `EmbeddingSFTConfig` from `mlx_tune`.

Output:
- Adapters → `workspaces/<domain>/adapters/`
- Metrics → `workspaces/<domain>/logs/training/training_metrics.json` (same schema as LM)

### Model config (`model_config.yaml`) — new `embedding` block

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
  base_model: mlx-community/ms-marco-MiniLM-L-6-v2   # pre-trained CE or custom
  max_seq_length: 512
  lora:
    rank: 8
    alpha: 8
    dropout: 0.0
```

The existing `base_model` / `lora` top-level block is only read for `lm` domains. Embedding domains read the `embedding` block.

### Training config (`training_config.yaml`) — new `embedding` block

```yaml
embedding:
  loss_type: infonce           # infonce | triplet
  temperature: 0.05            # used by infonce
  margin: 1.0                  # used by triplet
  normalize_embeddings: true
  anchor_column: anchor
  positive_column: positive
  negative_column: negative    # optional; omit for infonce with pairs only
```

### `commands/train.py`

Adds `embedding` as a valid `--method` value, dispatching to `src.training.embedding.run`.

---

## Section 4: Embedding Data Preparation

### `commands/prepare_embedding.py` (new)

CLI: `elixirtune prepare-embedding <domain> --mode <import|convert>`

**Mode: `import`** (bring your own data)

- `--data-file <path>` — JSON or JSONL with anchor/positive (and optionally negative) fields
- Validates column names against `embedding.*_column` config values
- `--val-split` (default 0.1) — splits into train/val
- Writes:
  - `workspaces/<domain>/processed/embedding_train.json`
  - `workspaces/<domain>/processed/embedding_val.json`

**Mode: `convert`** (auto-convert from existing domain seeds/generated content)

- Reads `workspaces/<domain>/seeds/approved.jsonl` and/or `generated/filtered.jsonl`
- Converts Q&A conversation pairs: question → anchor, answer → positive
- For triplet mode: negatives are drawn randomly from other records in the batch
- Same output location as import mode

### TUI (`tui/panels/embedding_training.py`)

- "Import data" button → triggers `import` mode (prompts for file path)
- "Convert from seeds" button → triggers `convert` mode (enabled only if seeds exist)
- "▶ Train" button → runs `commands/train.py --method embedding`
- "▶ Train cross-encoder" button → runs `commands/train.py --method cross-encoder` (enabled after bi-encoder is trained)
- Config form shows: base model, loss type, learning rate, iterations
- `LogView` and live metrics display (reuses existing widgets)

---

## Section 5: Embedding Evaluation

### `src/evaluation/embedding_evaluator.py` (new)

```python
def compute_similarity(anchor: str, candidates: list[str], model, tokenizer) -> list[float]
def recall_at_k(val_path: Path, model, tokenizer, k: list[int] = [1, 5, 10]) -> dict
def run_beir(dataset_name: str, model, tokenizer) -> dict  # NDCG@10, Recall@100
def rerank_with_cross_encoder(query: str, candidates: list[str], ce_model, tokenizer) -> list[float]
```

Uses `FastEmbeddingModel.for_inference(model)` before any inference call.

### BEIR Benchmark

- Evaluation panel includes a BEIR section: user selects one or more BEIR datasets (e.g. SciFact, NFCorpus, MSMARCO-small) from a dropdown
- Results: NDCG@10 and Recall@100 per dataset, displayed in a results table
- Requires the `beir` package (PyPI: `beir`). This is a heavy optional dependency; added to `requirements.txt` with a comment marking it optional. The eval panel gracefully disables the BEIR section with an install prompt if `beir` is not importable.

### Cross-Encoder Reranking

**Training stage** (`src/training/cross_encoder.py`, new):
- A second optional training step within an embedding domain
- Uses a BERT-like model with a classification head, fine-tuned on positive/negative pairs
- `commands/train.py --method cross-encoder` dispatches here
- Adapters saved to `workspaces/<domain>/ce_adapters/`

**Inference/eval** (`src/evaluation/embedding_evaluator.py`):
- In the eval panel, after bi-encoder retrieval, optionally rerank with the domain's cross-encoder (if trained) or a pre-trained cross-encoder model (configurable in model config)
- Reranked results shown side-by-side with bi-encoder results

### `commands/evaluate.py`

Extended with `--method embedding` and `--method cross-encoder`, dispatching to `embedding_evaluator`. The existing LM evaluation path is untouched.

### `tui/panels/embedding_eval.py` (new)

- **Cosine similarity probe**: user enters anchor + candidates; panel ranks by cosine similarity
- **Retrieval metrics**: Recall@1/5/10 over `embedding_val.json`
- **BEIR benchmark**: dataset selector, runs evaluation, displays NDCG@10 table
- **Reranking**: toggle to enable cross-encoder reranking over retrieval results

---

## File Map

| File | Change |
|---|---|
| Throughout | Rename `elixirlora` → `elixirtune` in strings/docs |
| `config/model_config.yaml` | Add `embedding:` block |
| `config/training_config.yaml` | Add `embedding:` block |
| `tui/domain.py` | Add `read_domain_type()`, extend `infer_status` |
| `tui/new_domain.py` | Add type selector step |
| `tui/app.py` | Panel routing by domain type; rename title |
| `tui/sidebar.py` | Domain type badge `[LM]` / `[EM]` |
| `tui/panels/embedding_training.py` | New — embedding data + training panel |
| `tui/panels/embedding_eval.py` | New — embedding eval panel |
| `src/training/embedding.py` | New — bi-encoder fine-tuning via mlx_tune |
| `src/training/cross_encoder.py` | New — cross-encoder fine-tuning |
| `src/evaluation/embedding_evaluator.py` | New — similarity, recall@k, BEIR, reranking |
| `commands/prepare_embedding.py` | New — import + convert data prep |
| `commands/train.py` | Add `embedding` and `cross-encoder` methods |
| `commands/evaluate.py` | Add `--method embedding` / `cross-encoder` |
| `requirements.txt` | Add `beir` |

---

## Out of Scope

- VLM, TTS, STT, JEPA training (future domain types, same extension pattern)
- Quantization of embedding adapters
- Distributed embedding training
