# ElixirLoRA — Synthetic Data Generation Pipeline (Design Spec)

**Version:** 1.0
**Date:** 2026-06-25
**Slice:** Synthetic data generation pipeline (first vertical slice of ElixirLoRA)
**First domain:** `code_review`
**Status:** Approved design — ready for implementation planning

---

## 1. Scope & Integration Boundary

This spec covers **only** the synthetic data generation pipeline — one vertical slice of
ElixirLoRA. It is **CLI-first**; the TUI is a later, separate slice.

### What we build
A pipeline that turns a domain description and/or a handful of seed examples into a
quality-filtered training dataset for the `code_review` domain.

### What we do NOT build (provided by the fork)
ElixirLoRA forks [`DidierRLopes/fine-tune-llm`](https://github.com/DidierRLopes/fine-tune-llm).
The fork already provides: training, evaluation, train/val/test splitting, chat-template
rendering, and `mlx_lm.lora` invocation. **We invent nothing the fork already does.**
Our pipeline's output is the **input** to the fork's existing data module.

### The handoff — one clean boundary
Our pipeline emits records in the fork's source format:

```json
{"conversation": [
  {"role": "user", "content": "<code/diff + review ask>"},
  {"role": "assistant", "content": "<the review>"}
]}
```

- Written where the fork's preprocessor reads source data.
- The code-reviewer **system prompt lives in `data_config.yaml`** (the fork's convention),
  **not** per-record.
- After our pipeline runs, the unmodified fork takes over. No glue code.

### Design principles (carried from project conventions)
- Follow the base repo's conventions; do not invent formats it doesn't expect.
- Minimum code that solves the problem; refinement loops and extras are opt-in, not default.
- Spend the complexity budget on **filtering** (the highest-value stage), not on generation tricks.

---

## 2. Workspace Model

Each domain is a **workspace** created via CLI. Workspace init asks
*"Do you have seed examples?"*:

- **Yes** → import them into the workspace.
- **No** → **bootstrap** a starter batch from a domain description + the teacher model.

Either path feeds a **mandatory human-curation gate**: seeds must be approved (edited/rejected)
before any expansion runs. This gate is **not skippable** — it is the cheapest insurance against
the #1 synthetic-data failure mode (training on unfiltered/unanchored data, where the student
inherits the teacher's generic blind spots). Bootstrapping breaks the blank-page problem; the
gate ensures the trusted seed set reflects the user's taste and edge cases before it anchors
thousands of generated examples.

---

## 3. Pipeline Architecture

### Stage flow
Every stage reads the previous stage's persisted output. All teacher interaction goes through
one shared client.

```
                       ┌─────────────────────────────────────────────┐
                       │  OpenAI-compatible teacher client (shared)   │
                       │  config: base_url + model + api_key          │
                       └─────────────────────────────────────────────┘
                              ▲          ▲           ▲         ▲
   init ──► seeds ──► CURATE ─┼─► GENERATE ─► REFINE[] ─► FILTER ─► ASSEMBLE ──► fork
   (import │  (bootstrap      │   few-shot     optional     4 checks   conversation
    or     │   or imported)   │   +opt CoT     toggles      (below)    records out
    boot)  │                  │
           └── human gate ────┘
```

- **Seed acquisition** — import existing examples *or* bootstrap from the domain description.
  Output: seed candidates.
- **Curation gate** — human approves/edits/rejects → trusted seed set. Not skippable.
- **Generate** — sample a few approved seeds as few-shot context; ask the teacher for new,
  diverse pairs. Optional CoT prompting (see §5). Output: raw candidates.
- **Refine[]** — ordered, each toggleable: `self_refine`, `critique_revise`. Uniform shape
  `(record) → teacher → record`. Empty list = pure few-shot expansion.
- **Filter** — schema-validate → dedup → LLM-as-judge cutoff → diversity quotas (see §6).
- **Assemble** — strip internal metadata, write `conversation` records where the fork reads them.

### Teacher abstraction
A single **OpenAI-compatible Chat Completions client**, driven entirely by config
(`base_url` + `model` + `api_key`). This one code path serves both worlds:

- **Local dev:** `base_url` → local `llama.cpp` server, `model` → Qwen3.6. Free, fast iteration.
- **Production:** user points `base_url`/`model`/`api_key` at any provider.

We do **not** build per-provider SDKs. The OpenAI-compatible API is the universal abstraction
(OpenAI, xAI/Grok, Together, Fireworks, local llama.cpp/Ollama, etc.). **Caveat:** Anthropic's
*native* API is not OpenAI-shaped — Claude works only via its OpenAI-compatibility endpoint or a
tiny adapter. This is a config awareness, not a blocker.

### Module layout
Additive to the fork's `src/data/` convention. We do not modify the fork's existing files.

```
src/data/synthetic/
  teacher.py        # OpenAI-compatible client, config-driven
  bootstrap.py      # domain description → starter seeds
  generate.py       # few-shot expansion (+ optional CoT)
  refine.py         # self_refine / critique_revise passes
  filter.py         # schema, dedup, judge, diversity
  assemble.py       # write conversation records
  pipeline.py       # orchestrates stages, honors config toggles
cli.py              # CLI commands (below)
```

### CLI surface (this slice)
- `elixirlora init <domain>` — create workspace; import-or-bootstrap seeds.
- `elixirlora curate <domain>` — review/edit/approve seeds.
- `elixirlora generate <domain>` — run generate → refine → filter → assemble.

Each stage persists its output to disk, making the pipeline inspectable and resumable
between stages (§7).

---

## 4. Data Contract & Workspace Layout

### Workspace directory layout
One directory per domain; each stage's output persisted for inspection/resume.

```
workspaces/code_review/
  domain.md              # domain description (bootstrap input)
  config.yaml            # per-domain config overrides
  seeds/
    candidates.jsonl     # raw bootstrapped/imported seeds (pre-curation)
    approved.jsonl       # post-curation trusted seeds
  generated/
    raw.jsonl            # generate stage output
    refined.jsonl        # after refine[] passes
    filtered.jsonl       # survivors of all 4 filters
  runs/
    <timestamp>/         # one dir per `generate` run (see §8)
      manifest.json
      rejected.jsonl
      stats.json
```

### Record formats by stage
Internal records carry metadata used by later stages; only the **final** assembled record is the
pure fork contract.

```jsonc
// seeds/approved.jsonl, generated/*.jsonl — internal, metadata-carrying
{
  "conversation": [
    {"role": "user", "content": "<code/diff + review ask>"},
    {"role": "assistant", "content": "<review>"}
  ],
  "meta": {                         // stripped before assemble
    "source": "bootstrap|fewshot|self_refine|critique_revise",
    "cot": "<reasoning, never trained on>",
    "judge_score": 4,
    "category": "bug",
    "seed_ids": ["s3", "s7"]        // provenance for diversity/debug
  }
}
```

```jsonc
// final output to the fork — meta dropped, pure contract
{"conversation": [
  {"role": "user", "content": "..."},
  {"role": "assistant", "content": "..."}
]}
```

The system prompt is **not** in the record — it lives in the fork's `data_config.yaml`.

---

## 5. Generation & Refinement

### Generation
- **Few-shot expansion** is the core: sample `fewshot_k` approved seeds as in-context examples,
  ask the teacher for new, diverse code-review pairs.
- **CoT (optional, prompting technique — not a refinement pass):** ask the teacher to reason
  step-by-step before producing the review. The reasoning is captured in `meta.cot` for
  debugging but **stripped from the trained `assistant` content** — the model should learn to
  produce the *review*, not the reasoning.

### Refinement (all available day one, all toggleable)
Refinement is a separate stage from generation. Each strategy is **one pluggable pass with a
config toggle**, sharing a uniform shape `(record) → teacher call → improved record`:

- `self_refine` — teacher critiques and improves its own output.
- `critique_revise` — teacher critiques, then revises against the critique.

`refine.passes: []` → pure few-shot expansion (loops ignored). Listing passes runs them in order.
This keeps "all available day one" cheap: adding a strategy is one uniform pass, not a bespoke
subsystem.

**Rationale:** the biggest quality lever is filtering, not generation loops. Refinement loops
multiply teacher calls and you cannot tell whether they help until there is a baseline dataset to
compare against — so they exist but default off.

---

## 6. Filtering (Highest-Value Stage)

Run in order; cheapest/most-deterministic first.

1. **Schema/format validation** — drop anything malformed or not matching the `conversation`
   contract; enforce `max_tokens`. Deterministic.
2. **Dedup** — exact-match hash, plus near-duplicate via embedding similarity above a threshold,
   so the teacher cannot flood the set with the same review reworded.
3. **LLM-as-judge** — score each candidate (e.g. 1–5) and keep those at/above a cutoff. Highest
   quality signal; costs teacher calls.
4. **Diversity control** — track coverage across issue-category quotas so the set is not 90%
   "missing null check."

Every dropped record is written to the run's `rejected.jsonl` **with a reason**, for auditability.

---

## 7. Error Handling & Resumability

### Teacher-call failures (the main failure surface)
- **Transient** (timeout, 429, 5xx) → retry with exponential backoff, capped attempts. On
  exhaustion, **skip that record, log it**, continue. One bad call must never kill a full run.
- **Malformed teacher output** (unparseable structure/CoT) → treated as a generation *miss*:
  record dropped to `rejected.jsonl` with reason, never crashed on. The teacher is
  non-deterministic; the pipeline assumes some outputs are junk.
- **Auth/config errors** (bad `api_key`, unreachable `base_url`) → **fail fast at startup** with a
  clear message, before spending any calls.

### Cost/safety guards (matter once `base_url` points at a paid API)
- `--dry-run`: do N calls and report projected total calls before a full run.
- `target_size` is a hard ceiling; retries count toward a max-calls budget so a misconfig cannot
  run away.

### Resumability
- Every stage writes its output file → a run **resumes from the last completed stage** rather than
  restarting.
- Within `generate`, progress is checkpointed by **appending** to `raw.jsonl` → an interrupted
  generation continues toward `target_size` instead of regenerating.
- Each stage is **idempotent** on its inputs.
- The **curation gate is enforced**: `generate` refuses to run if `seeds/approved.jsonl` is absent.

---

## 8. Traceability (Generation Provenance)

The fork provides **no** run tracking — only a `logs/` folder and YAML configs, with no run
manifests or per-run config snapshots. Moreover, generation provenance is **structurally ours**:
the fork has no concept of synthetic generation, so it cannot record what teacher/seeds/strategy
produced a dataset. We own this.

Each `generate` run writes a local-JSON **run manifest** making the dataset reproducible and
auditable:

```
workspaces/code_review/runs/2026-06-25T14-30/
  manifest.json     # resolved config snapshot, teacher (base_url + model),
                    # seed-set hash, per-stage in→out counts,
                    # judge-score distribution, git sha, timestamp
  rejected.jsonl    # dropped records + reasons
  stats.json        # diversity coverage, counts, judge-score distribution
```

Deliberate choices:
- **Local JSON, no external deps.** Consistent with the project's stated stack
  ("Local JSON logs + optional MLflow"). MLflow / W&B stay deferred (YAGNI) until cross-run
  dashboards are actually wanted.
- **Mirrors the fork's lightweight ethos** (`logs/` + YAML) while adding the one thing the fork
  lacks: *"this exact dataset came from this config + teacher + seeds."*

---

## 9. Configuration Model

`config/defaults.yaml` is the **single source of truth**: every knob present and exposed, set to
its default value. A per-domain `config.yaml` overrides only what it needs; the loader
**deep-merges domain over defaults**.

- **Values live in config**, not hardcoded in code → no magic numbers, no drift.
- The **development doc explains each knob and suggests values** for users who are unsure; it does
  not hold an authoritative copy of the values (that would drift).

```yaml
# config/defaults.yaml — full surface, every knob exposed
teacher:
  base_url: http://localhost:8080/v1   # dev: llama.cpp + Qwen3.6
  model: qwen3.6
  api_key: ""
  temperature: 0.8

bootstrap:
  starter_count: 20

generate:
  target_size: 2000
  fewshot_k: 4                    # seeds sampled into context
  cot: true                       # reason first, strip from output

refine:                           # ordered; empty list = skip all
  passes: []                      # e.g. [self_refine, critique_revise]
  self_refine:     {model: qwen3.6}
  critique_revise: {model: qwen3.6}

filter:
  schema:    {max_tokens: 2048}
  dedup:     {embedding_model: <name>, similarity_threshold: 0.92}
  judge:     {model: qwen3.6, score_cutoff: 4, prompt_template: <ref>}
  diversity: {quotas: {bug: 0.4, style: 0.3, perf: 0.3}}
```

```yaml
# workspaces/code_review/config.yaml — overrides only
teacher: {base_url: https://api.provider.com/v1, model: <prod>, api_key: ${KEY}}
generate: {target_size: 5000}
refine: {passes: [critique_revise]}
```

---

## 10. Testing Strategy

The pipeline is teacher-driven, so the key move is a **`FakeTeacher`** — deterministic canned
responses behind the same OpenAI-compatible interface.

- **Unit** — each stage against `FakeTeacher`. Generate produces well-formed records; refine
  transforms correctly. **Filters get the most coverage** (highest-value stage): golden tests for
  schema-reject, exact + near-dup dedup at the threshold, judge cutoff, diversity quotas.
- **Contract test** — assembled output exactly matches the fork's `conversation` shape and loads
  through its preprocessor. Guards the single integration boundary.
- **Resume test** — kill mid-`generate`, rerun, assert it continues toward `target_size` rather
  than restarting.
- **Integration smoke test** — one real end-to-end run against **local llama.cpp + Qwen3.6**
  producing a handful of records. Cheap because it is local.

---

## 11. Out of Scope (Future Slices)
- TUI dashboard (presentation layer over this pipeline).
- Training / evaluation / deployment slices (largely provided by the fork).
- Additional domains beyond `code_review`.
- Multi-teacher mixing, advanced self-improvement loops, MLflow/W&B tracking, web UI.
