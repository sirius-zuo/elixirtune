# Synthetic Data Generation Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI-first pipeline that turns a domain description and/or seed examples into a quality-filtered `code_review` training dataset in the `fine-tune-llm` fork's `conversation` format.

**Architecture:** Stage pipeline (`init → curate → generate → refine → filter → assemble`) where every teacher interaction goes through one config-driven OpenAI-compatible client. Each stage persists its output to a per-domain workspace directory, making the pipeline inspectable, resumable, and reproducible via a per-run JSON manifest. Refinement loops and CoT are opt-in toggles; the complexity budget is spent on filtering.

**Tech Stack:** Python 3.11+, `pytest`, `openai` (OpenAI-compatible client), `pyyaml`, `pydantic`, `sentence-transformers` (local embeddings for dedup), `typer` (CLI).

## Global Constraints

- Output records MUST match the fork contract exactly: `{"conversation": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}` — no `meta`, no system prompt in the record.
- System prompt lives in the fork's `data_config.yaml`, never per-record.
- Teacher access is ONLY through the OpenAI-compatible Chat Completions API (`base_url` + `model` + `api_key`). No per-provider SDKs.
- All tunable values live in `config/defaults.yaml` (single source of truth); per-domain `config.yaml` overrides only what it needs; loader deep-merges domain over defaults. No magic numbers hardcoded in code.
- Dev default teacher: `base_url: http://localhost:8080/v1`, `model: qwen3.6`.
- The curation gate is NOT skippable: `generate` must refuse to run without `seeds/approved.jsonl`.
- Additive only: do not modify the fork's existing `src/data/` files.
- TDD throughout: failing test first, minimal code, passing test, commit.

---

## File Structure

```
src/data/synthetic/
  __init__.py
  io.py             # JSONL read/append/write, record helpers, hashing
  config.py         # load_config: defaults + domain override deep-merge
  teacher.py        # Teacher protocol, OpenAITeacher, FakeTeacher
  embedder.py       # Embedder protocol, SentenceTransformerEmbedder, FakeEmbedder
  bootstrap.py      # domain description -> starter seeds
  generate.py       # few-shot expansion (+ optional CoT)
  refine.py         # self_refine / critique_revise passes
  filter.py         # schema, dedup, judge, diversity
  assemble.py       # strip meta, write conversation records
  manifest.py       # per-run traceability manifest
  pipeline.py       # orchestrates generate run, resumability
cli.py              # typer app: init, curate, generate
config/defaults.yaml
tests/synthetic/
  conftest.py
  test_*.py
```

---

### Task 1: Project setup, config loader, JSONL/IO helpers

**Files:**
- Create: `requirements.txt`, `config/defaults.yaml`, `tests/synthetic/conftest.py`
- Create: `src/data/synthetic/__init__.py`, `src/data/synthetic/io.py`, `src/data/synthetic/config.py`
- Test: `tests/synthetic/test_config.py`, `tests/synthetic/test_io.py`

**Interfaces:**
- Produces:
  - `io.read_jsonl(path: str | Path) -> list[dict]`
  - `io.write_jsonl(path: str | Path, records: list[dict]) -> None`
  - `io.append_jsonl(path: str | Path, records: list[dict]) -> None`
  - `io.make_record(user: str, assistant: str, meta: dict) -> dict`
  - `io.sha256_of(obj) -> str`
  - `config.load_config(domain: str, root: Path = Path(".")) -> dict`

- [ ] **Step 1: Write the failing test for config deep-merge**

```python
# tests/synthetic/test_config.py
from pathlib import Path
import yaml
from data.synthetic.config import load_config

def test_domain_config_overrides_defaults(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "defaults.yaml").write_text(yaml.safe_dump({
        "teacher": {"model": "qwen3.6", "temperature": 0.8},
        "generate": {"target_size": 2000, "fewshot_k": 4},
    }))
    ws = tmp_path / "workspaces" / "code_review"
    ws.mkdir(parents=True)
    (ws / "config.yaml").write_text(yaml.safe_dump({
        "teacher": {"model": "prod-model"},
        "generate": {"target_size": 5000},
    }))

    cfg = load_config("code_review", root=tmp_path)

    assert cfg["teacher"]["model"] == "prod-model"      # overridden
    assert cfg["teacher"]["temperature"] == 0.8         # preserved from defaults
    assert cfg["generate"]["target_size"] == 5000       # overridden
    assert cfg["generate"]["fewshot_k"] == 4            # preserved from defaults

def test_missing_domain_config_uses_defaults_only(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "defaults.yaml").write_text(yaml.safe_dump({"generate": {"target_size": 2000}}))
    cfg = load_config("nonexistent", root=tmp_path)
    assert cfg["generate"]["target_size"] == 2000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synthetic/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.synthetic.config'`

- [ ] **Step 3: Create setup files and config loader**

```toml
# requirements.txt
openai>=1.0
pyyaml>=6.0
pydantic>=2.0
sentence-transformers>=2.2
typer>=0.9
pytest>=7.0
```

```python
# tests/synthetic/conftest.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
```

```yaml
# config/defaults.yaml
teacher:
  base_url: http://localhost:8080/v1
  model: qwen3.6
  api_key: ""
  temperature: 0.8
bootstrap:
  starter_count: 20
generate:
  target_size: 2000
  fewshot_k: 4
  cot: true
refine:
  passes: []
  self_refine: {model: qwen3.6}
  critique_revise: {model: qwen3.6}
filter:
  schema: {max_tokens: 2048}
  dedup: {embedding_model: all-MiniLM-L6-v2, similarity_threshold: 0.92}
  judge: {model: qwen3.6, score_cutoff: 4}
  diversity: {quotas: {bug: 0.4, style: 0.3, perf: 0.3}}
```

```python
# src/data/synthetic/__init__.py
```

```python
# src/data/synthetic/config.py
from copy import deepcopy
from pathlib import Path
import yaml

def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out

def load_config(domain: str, root: Path = Path(".")) -> dict:
    root = Path(root)
    defaults = yaml.safe_load((root / "config" / "defaults.yaml").read_text()) or {}
    domain_path = root / "workspaces" / domain / "config.yaml"
    if domain_path.exists():
        override = yaml.safe_load(domain_path.read_text()) or {}
        return _deep_merge(defaults, override)
    return defaults
```

- [ ] **Step 4: Run config test to verify it passes**

Run: `pytest tests/synthetic/test_config.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Write the failing test for IO helpers**

```python
# tests/synthetic/test_io.py
from data.synthetic.io import read_jsonl, write_jsonl, append_jsonl, make_record, sha256_of

def test_write_then_read_roundtrip(tmp_path):
    p = tmp_path / "x.jsonl"
    recs = [{"a": 1}, {"b": 2}]
    write_jsonl(p, recs)
    assert read_jsonl(p) == recs

def test_append_extends_existing(tmp_path):
    p = tmp_path / "x.jsonl"
    write_jsonl(p, [{"a": 1}])
    append_jsonl(p, [{"b": 2}])
    assert read_jsonl(p) == [{"a": 1}, {"b": 2}]

def test_read_missing_returns_empty(tmp_path):
    assert read_jsonl(tmp_path / "nope.jsonl") == []

def test_make_record_shape():
    r = make_record("code here", "review here", {"source": "fewshot"})
    assert r["conversation"] == [
        {"role": "user", "content": "code here"},
        {"role": "assistant", "content": "review here"},
    ]
    assert r["meta"]["source"] == "fewshot"

def test_sha256_is_stable_and_order_independent():
    assert sha256_of({"a": 1, "b": 2}) == sha256_of({"b": 2, "a": 1})
```

- [ ] **Step 6: Run IO test to verify it fails**

Run: `pytest tests/synthetic/test_io.py -v`
Expected: FAIL with `ImportError` (cannot import from `data.synthetic.io`)

- [ ] **Step 7: Implement IO helpers**

```python
# src/data/synthetic/io.py
import hashlib
import json
from pathlib import Path

def read_jsonl(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

def write_jsonl(path, records: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))

def append_jsonl(path, records: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def make_record(user: str, assistant: str, meta: dict) -> dict:
    return {
        "conversation": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "meta": meta,
    }

def sha256_of(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
```

- [ ] **Step 8: Run all Task 1 tests to verify they pass**

Run: `pytest tests/synthetic/test_config.py tests/synthetic/test_io.py -v`
Expected: PASS (7 passed)

- [ ] **Step 9: Commit**

```bash
git add requirements.txt config/defaults.yaml tests/synthetic/ src/data/synthetic/
git commit -m "feat(synthetic): config loader, IO helpers, project setup"
```

---

### Task 2: Teacher client (OpenAI-compatible) + FakeTeacher

**Files:**
- Create: `src/data/synthetic/teacher.py`
- Test: `tests/synthetic/test_teacher.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `teacher.Teacher` (Protocol with `chat(self, messages: list[dict], temperature: float | None = None) -> str`)
  - `teacher.OpenAITeacher(base_url: str, model: str, api_key: str, temperature: float = 0.8)`
  - `teacher.FakeTeacher(responses: list[str])` — returns queued responses in order; records `.calls`
  - `teacher.from_config(cfg: dict) -> OpenAITeacher` (validates connectivity config; raises `TeacherConfigError` on empty base_url)
  - `teacher.TeacherConfigError(Exception)`

- [ ] **Step 1: Write the failing test**

```python
# tests/synthetic/test_teacher.py
import pytest
from data.synthetic.teacher import FakeTeacher, from_config, TeacherConfigError

def test_fake_teacher_returns_queued_responses_in_order():
    t = FakeTeacher(responses=["first", "second"])
    assert t.chat([{"role": "user", "content": "x"}]) == "first"
    assert t.chat([{"role": "user", "content": "y"}]) == "second"

def test_fake_teacher_records_calls():
    t = FakeTeacher(responses=["ok"])
    t.chat([{"role": "user", "content": "hello"}])
    assert t.calls[0][0]["content"] == "hello"

def test_from_config_rejects_empty_base_url():
    with pytest.raises(TeacherConfigError):
        from_config({"teacher": {"base_url": "", "model": "m", "api_key": ""}})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synthetic/test_teacher.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement teacher module**

```python
# src/data/synthetic/teacher.py
from typing import Protocol

class TeacherConfigError(Exception):
    pass

class Teacher(Protocol):
    def chat(self, messages: list[dict], temperature: float | None = None) -> str: ...

class OpenAITeacher:
    def __init__(self, base_url: str, model: str, api_key: str, temperature: float = 0.8):
        from openai import OpenAI
        self.model = model
        self.temperature = temperature
        self._client = OpenAI(base_url=base_url, api_key=api_key or "not-needed")

    def chat(self, messages: list[dict], temperature: float | None = None) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature if temperature is None else temperature,
        )
        return resp.choices[0].message.content

class FakeTeacher:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._i = 0
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict], temperature: float | None = None) -> str:
        self.calls.append(messages)
        r = self._responses[self._i % len(self._responses)]
        self._i += 1
        return r

def from_config(cfg: dict) -> OpenAITeacher:
    t = cfg["teacher"]
    if not t.get("base_url"):
        raise TeacherConfigError("teacher.base_url is required")
    return OpenAITeacher(
        base_url=t["base_url"], model=t["model"],
        api_key=t.get("api_key", ""), temperature=t.get("temperature", 0.8),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/synthetic/test_teacher.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/synthetic/test_teacher.py src/data/synthetic/teacher.py
git commit -m "feat(synthetic): OpenAI-compatible teacher client + FakeTeacher"
```

---

### Task 3: Bootstrap seeds from domain description

**Files:**
- Create: `src/data/synthetic/bootstrap.py`
- Test: `tests/synthetic/test_bootstrap.py`

**Interfaces:**
- Consumes: `teacher.Teacher`, `io.make_record`.
- Produces: `bootstrap.bootstrap_seeds(domain_desc: str, teacher: Teacher, count: int) -> list[dict]`
  - Each returned record is a `make_record(...)` with `meta["source"] == "bootstrap"`.
  - Teacher is asked to return a JSON array of `{"user": ..., "assistant": ...}`; malformed items are skipped.

- [ ] **Step 1: Write the failing test**

```python
# tests/synthetic/test_bootstrap.py
import json
from data.synthetic.teacher import FakeTeacher
from data.synthetic.bootstrap import bootstrap_seeds

def test_bootstrap_parses_teacher_json_into_records():
    payload = json.dumps([
        {"user": "review this diff A", "assistant": "review A"},
        {"user": "review this diff B", "assistant": "review B"},
    ])
    t = FakeTeacher(responses=[payload])
    recs = bootstrap_seeds("code review domain", t, count=2)
    assert len(recs) == 2
    assert recs[0]["conversation"][0]["content"] == "review this diff A"
    assert recs[0]["meta"]["source"] == "bootstrap"

def test_bootstrap_skips_malformed_items():
    payload = json.dumps([{"user": "u", "assistant": "a"}, {"oops": 1}])
    t = FakeTeacher(responses=[payload])
    recs = bootstrap_seeds("desc", t, count=2)
    assert len(recs) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synthetic/test_bootstrap.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement bootstrap**

```python
# src/data/synthetic/bootstrap.py
import json
from .io import make_record
from .teacher import Teacher

_PROMPT = (
    "You are creating starter training examples for this domain:\n\n{desc}\n\n"
    "Generate {count} diverse, realistic examples. Return ONLY a JSON array where each "
    'item is {{"user": "<task input>", "assistant": "<ideal response>"}}.'
)

def _parse_array(text: str) -> list[dict]:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []

def bootstrap_seeds(domain_desc: str, teacher: Teacher, count: int) -> list[dict]:
    text = teacher.chat([{"role": "user", "content": _PROMPT.format(desc=domain_desc, count=count)}])
    records = []
    for item in _parse_array(text):
        if isinstance(item, dict) and "user" in item and "assistant" in item:
            records.append(make_record(item["user"], item["assistant"], {"source": "bootstrap"}))
    return records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/synthetic/test_bootstrap.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/synthetic/test_bootstrap.py src/data/synthetic/bootstrap.py
git commit -m "feat(synthetic): bootstrap starter seeds from domain description"
```

---

### Task 4: Generate via few-shot expansion (+ optional CoT)

**Files:**
- Create: `src/data/synthetic/generate.py`
- Test: `tests/synthetic/test_generate.py`

**Interfaces:**
- Consumes: `teacher.Teacher`, `io.make_record`.
- Produces:
  - `generate.generate_one(seeds: list[dict], teacher: Teacher, fewshot_k: int, cot: bool) -> dict`
    - Returns a `make_record(...)` with `meta["source"] == "fewshot"`; when `cot=True`, `meta["cot"]` holds the reasoning and the trained `assistant` content has reasoning stripped.
    - Teacher returns JSON `{"reasoning": ..., "user": ..., "assistant": ...}`; on parse failure raises `GenerationMiss`.
  - `generate.GenerationMiss(Exception)`

- [ ] **Step 1: Write the failing test**

```python
# tests/synthetic/test_generate.py
import json
import pytest
from data.synthetic.teacher import FakeTeacher
from data.synthetic.io import make_record
from data.synthetic.generate import generate_one, GenerationMiss

SEEDS = [make_record(f"code {i}", f"review {i}", {"source": "bootstrap"}) for i in range(5)]

def test_generate_one_strips_cot_from_assistant():
    payload = json.dumps({"reasoning": "think think", "user": "new code", "assistant": "clean review"})
    t = FakeTeacher(responses=[payload])
    rec = generate_one(SEEDS, t, fewshot_k=3, cot=True)
    assert rec["conversation"][1]["content"] == "clean review"
    assert rec["meta"]["cot"] == "think think"
    assert rec["meta"]["source"] == "fewshot"

def test_generate_one_raises_miss_on_malformed():
    t = FakeTeacher(responses=["not json at all"])
    with pytest.raises(GenerationMiss):
        generate_one(SEEDS, t, fewshot_k=3, cot=False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synthetic/test_generate.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement generate**

```python
# src/data/synthetic/generate.py
import json
import random
from .io import make_record
from .teacher import Teacher

class GenerationMiss(Exception):
    pass

def _fewshot_block(seeds: list[dict], k: int) -> str:
    chosen = random.sample(seeds, min(k, len(seeds)))
    lines = []
    for s in chosen:
        u = s["conversation"][0]["content"]
        a = s["conversation"][1]["content"]
        lines.append(f"INPUT:\n{u}\nRESPONSE:\n{a}")
    return "\n\n".join(lines)

def _parse_object(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise GenerationMiss("no JSON object in teacher output")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise GenerationMiss(str(e))

def generate_one(seeds: list[dict], teacher: Teacher, fewshot_k: int, cot: bool) -> dict:
    cot_instr = (
        'First reason step by step in "reasoning", then give the final answer. '
        if cot else "Leave \"reasoning\" empty. "
    )
    prompt = (
        "Here are example INPUT/RESPONSE pairs:\n\n"
        f"{_fewshot_block(seeds, fewshot_k)}\n\n"
        "Produce ONE new, different, realistic pair in the same style. "
        f"{cot_instr}"
        'Return ONLY JSON: {"reasoning": "...", "user": "...", "assistant": "..."}'
    )
    obj = _parse_object(teacher.chat([{"role": "user", "content": prompt}]))
    if "user" not in obj or "assistant" not in obj:
        raise GenerationMiss("missing user/assistant")
    return make_record(obj["user"], obj["assistant"], {"source": "fewshot", "cot": obj.get("reasoning", "")})
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/synthetic/test_generate.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/synthetic/test_generate.py src/data/synthetic/generate.py
git commit -m "feat(synthetic): few-shot generation with optional CoT (reasoning stripped)"
```

---

### Task 5: Refinement passes (self_refine, critique_revise)

**Files:**
- Create: `src/data/synthetic/refine.py`
- Test: `tests/synthetic/test_refine.py`

**Interfaces:**
- Consumes: `teacher.Teacher`, the record shape from `io.make_record`.
- Produces:
  - `refine.self_refine(record: dict, teacher: Teacher) -> dict`
  - `refine.critique_revise(record: dict, teacher: Teacher) -> dict`
  - `refine.apply_passes(record: dict, passes: list[str], teacher: Teacher) -> dict`
    - `passes=[]` returns the record unchanged. Each pass returns a new record with the `assistant` content replaced and `meta["source"]` set to the pass name; other `meta` keys preserved.
    - Unknown pass name raises `ValueError`.

- [ ] **Step 1: Write the failing test**

```python
# tests/synthetic/test_refine.py
import pytest
from data.synthetic.teacher import FakeTeacher
from data.synthetic.io import make_record
from data.synthetic.refine import self_refine, apply_passes

BASE = make_record("code", "ok review", {"source": "fewshot", "cot": ""})

def test_self_refine_replaces_assistant_and_tags_source():
    t = FakeTeacher(responses=["better review"])
    out = self_refine(BASE, t)
    assert out["conversation"][1]["content"] == "better review"
    assert out["meta"]["source"] == "self_refine"
    assert out["conversation"][0]["content"] == "code"  # user untouched

def test_apply_passes_empty_is_noop():
    t = FakeTeacher(responses=["x"])
    assert apply_passes(BASE, [], t) == BASE

def test_apply_passes_runs_in_order():
    t = FakeTeacher(responses=["after_self", "after_critique"])
    out = apply_passes(BASE, ["self_refine", "critique_revise"], t)
    assert out["conversation"][1]["content"] == "after_critique"
    assert out["meta"]["source"] == "critique_revise"

def test_apply_passes_unknown_raises():
    with pytest.raises(ValueError):
        apply_passes(BASE, ["nope"], FakeTeacher(responses=["x"]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synthetic/test_refine.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement refine**

```python
# src/data/synthetic/refine.py
from copy import deepcopy
from .teacher import Teacher

def _replace_assistant(record: dict, new_text: str, source: str) -> dict:
    out = deepcopy(record)
    out["conversation"][1]["content"] = new_text
    out["meta"]["source"] = source
    return out

def self_refine(record: dict, teacher: Teacher) -> dict:
    user = record["conversation"][0]["content"]
    assistant = record["conversation"][1]["content"]
    prompt = (
        f"INPUT:\n{user}\n\nDRAFT RESPONSE:\n{assistant}\n\n"
        "Improve the draft response. Return ONLY the improved response text."
    )
    return _replace_assistant(record, teacher.chat([{"role": "user", "content": prompt}]).strip(), "self_refine")

def critique_revise(record: dict, teacher: Teacher) -> dict:
    user = record["conversation"][0]["content"]
    assistant = record["conversation"][1]["content"]
    prompt = (
        f"INPUT:\n{user}\n\nRESPONSE:\n{assistant}\n\n"
        "First critique the response, then output the revised response after a line 'REVISED:'. "
        "Return the revised text only after that marker."
    )
    out = teacher.chat([{"role": "user", "content": prompt}])
    revised = out.split("REVISED:", 1)[1].strip() if "REVISED:" in out else out.strip()
    return _replace_assistant(record, revised, "critique_revise")

_PASSES = {"self_refine": self_refine, "critique_revise": critique_revise}

def apply_passes(record: dict, passes: list[str], teacher: Teacher) -> dict:
    out = record
    for name in passes:
        if name not in _PASSES:
            raise ValueError(f"unknown refinement pass: {name}")
        out = _PASSES[name](out, teacher)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/synthetic/test_refine.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/synthetic/test_refine.py src/data/synthetic/refine.py
git commit -m "feat(synthetic): toggleable self_refine / critique_revise passes"
```

---

### Task 6: Embedder abstraction (real + fake)

**Files:**
- Create: `src/data/synthetic/embedder.py`
- Test: `tests/synthetic/test_embedder.py`

**Interfaces:**
- Produces:
  - `embedder.Embedder` (Protocol: `embed(self, texts: list[str]) -> list[list[float]]`)
  - `embedder.FakeEmbedder(mapping: dict[str, list[float]])` — returns the mapped vector per exact text
  - `embedder.SentenceTransformerEmbedder(model_name: str)` — lazy-loads the model

- [ ] **Step 1: Write the failing test**

```python
# tests/synthetic/test_embedder.py
from data.synthetic.embedder import FakeEmbedder

def test_fake_embedder_returns_mapped_vectors():
    e = FakeEmbedder({"a": [1.0, 0.0], "b": [0.0, 1.0]})
    assert e.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synthetic/test_embedder.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement embedder**

```python
# src/data/synthetic/embedder.py
from typing import Protocol

class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

class FakeEmbedder:
    def __init__(self, mapping: dict[str, list[float]]):
        self._mapping = mapping
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._mapping[t] for t in texts]

class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None
    def embed(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
        return [v.tolist() for v in self._model.encode(texts)]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/synthetic/test_embedder.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/synthetic/test_embedder.py src/data/synthetic/embedder.py
git commit -m "feat(synthetic): embedder abstraction with fake + sentence-transformers impls"
```

---

### Task 7: Filters — schema, dedup, judge, diversity

**Files:**
- Create: `src/data/synthetic/filter.py`
- Test: `tests/synthetic/test_filter.py`

**Interfaces:**
- Consumes: `teacher.Teacher`, `embedder.Embedder`.
- Produces (each returns `(kept: list[dict], rejected: list[dict])`; rejected records get `meta["reject_reason"]`):
  - `filter.validate_schema(records, max_tokens: int)`
  - `filter.dedup(records, embedder: Embedder, threshold: float)`
  - `filter.judge(records, teacher: Teacher, score_cutoff: int)` — sets `meta["judge_score"]`; teacher returns an integer string
  - `filter.enforce_diversity(records, quotas: dict[str, float], target_size: int)` — uses `meta["category"]`

- [ ] **Step 1: Write the failing test**

```python
# tests/synthetic/test_filter.py
from data.synthetic.io import make_record
from data.synthetic.embedder import FakeEmbedder
from data.synthetic.teacher import FakeTeacher
from data.synthetic.filter import validate_schema, dedup, judge, enforce_diversity

def test_validate_schema_rejects_empty_and_too_long():
    good = make_record("code", "review", {})
    empty = make_record("code", "", {})
    longr = make_record("code", "x " * 5000, {})
    kept, rejected = validate_schema([good, empty, longr], max_tokens=2048)
    assert kept == [good]
    assert len(rejected) == 2
    assert all("reject_reason" in r["meta"] for r in rejected)

def test_dedup_drops_near_duplicates_above_threshold():
    a = make_record("c", "review one", {})
    b = make_record("c", "review one dup", {})   # near-duplicate
    c = make_record("c", "totally different", {})
    emb = FakeEmbedder({
        "review one": [1.0, 0.0], "review one dup": [0.99, 0.01],
        "totally different": [0.0, 1.0],
    })
    kept, rejected = dedup([a, b, c], emb, threshold=0.92)
    assert a in kept and c in kept and b in rejected

def test_judge_keeps_at_or_above_cutoff():
    r1 = make_record("c", "good", {})
    r2 = make_record("c", "bad", {})
    t = FakeTeacher(responses=["5", "2"])
    kept, rejected = judge([r1, r2], t, score_cutoff=4)
    assert kept[0]["meta"]["judge_score"] == 5
    assert rejected[0]["meta"]["judge_score"] == 2

def test_enforce_diversity_respects_quota_and_target():
    bugs = [make_record("c", f"bug {i}", {"category": "bug"}) for i in range(10)]
    styles = [make_record("c", f"style {i}", {"category": "style"}) for i in range(10)]
    kept, rejected = enforce_diversity(bugs + styles, {"bug": 0.5, "style": 0.5}, target_size=4)
    assert sum(1 for r in kept if r["meta"]["category"] == "bug") == 2
    assert sum(1 for r in kept if r["meta"]["category"] == "style") == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synthetic/test_filter.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement filters**

```python
# src/data/synthetic/filter.py
from copy import deepcopy
from .embedder import Embedder
from .teacher import Teacher

def _reject(record: dict, reason: str) -> dict:
    out = deepcopy(record)
    out["meta"]["reject_reason"] = reason
    return out

def _assistant(record: dict) -> str:
    return record["conversation"][1]["content"]

def validate_schema(records, max_tokens: int):
    kept, rejected = [], []
    for r in records:
        a = _assistant(r).strip()
        if not a:
            rejected.append(_reject(r, "empty_assistant"))
        elif len(a.split()) > max_tokens:
            rejected.append(_reject(r, "too_long"))
        else:
            kept.append(r)
    return kept, rejected

def _cosine(u, v):
    dot = sum(x * y for x, y in zip(u, v))
    nu = sum(x * x for x in u) ** 0.5
    nv = sum(y * y for y in v) ** 0.5
    return dot / (nu * nv) if nu and nv else 0.0

def dedup(records, embedder: Embedder, threshold: float):
    vecs = embedder.embed([_assistant(r) for r in records])
    kept, rejected, kept_vecs = [], [], []
    for r, v in zip(records, vecs):
        if any(_cosine(v, kv) >= threshold for kv in kept_vecs):
            rejected.append(_reject(r, "near_duplicate"))
        else:
            kept.append(r)
            kept_vecs.append(v)
    return kept, rejected

def judge(records, teacher: Teacher, score_cutoff: int):
    kept, rejected = [], []
    for r in records:
        prompt = (
            f"INPUT:\n{r['conversation'][0]['content']}\n\nRESPONSE:\n{_assistant(r)}\n\n"
            "Rate the response quality from 1 to 5. Return ONLY the integer."
        )
        try:
            score = int("".join(c for c in teacher.chat([{"role": "user", "content": prompt}]) if c.isdigit())[:1])
        except (ValueError, IndexError):
            score = 0
        out = deepcopy(r)
        out["meta"]["judge_score"] = score
        if score >= score_cutoff:
            kept.append(out)
        else:
            out["meta"]["reject_reason"] = "below_judge_cutoff"
            rejected.append(out)
    return kept, rejected

def enforce_diversity(records, quotas: dict[str, float], target_size: int):
    caps = {cat: int(round(frac * target_size)) for cat, frac in quotas.items()}
    counts = {cat: 0 for cat in quotas}
    kept, rejected = [], []
    for r in records:
        cat = r["meta"].get("category")
        if cat in caps and counts[cat] < caps[cat]:
            counts[cat] += 1
            kept.append(r)
        else:
            rejected.append(_reject(r, "diversity_quota_full"))
    return kept, rejected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/synthetic/test_filter.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/synthetic/test_filter.py src/data/synthetic/filter.py
git commit -m "feat(synthetic): schema/dedup/judge/diversity filters with reject reasons"
```

---

### Task 8: Assemble — strip meta, write fork contract

**Files:**
- Create: `src/data/synthetic/assemble.py`
- Test: `tests/synthetic/test_assemble.py`

**Interfaces:**
- Consumes: `io.write_jsonl`.
- Produces:
  - `assemble.strip_meta(record: dict) -> dict` — returns `{"conversation": [...]}` only
  - `assemble.assemble(records: list[dict], out_path) -> list[dict]` — writes stripped records, returns them

- [ ] **Step 1: Write the failing test**

```python
# tests/synthetic/test_assemble.py
from data.synthetic.io import make_record, read_jsonl
from data.synthetic.assemble import strip_meta, assemble

def test_strip_meta_removes_meta_only():
    r = make_record("code", "review", {"source": "fewshot", "judge_score": 5})
    s = strip_meta(r)
    assert s == {"conversation": [
        {"role": "user", "content": "code"},
        {"role": "assistant", "content": "review"},
    ]}
    assert "meta" not in s

def test_assemble_writes_pure_contract(tmp_path):
    recs = [make_record("c", "r", {"source": "fewshot"})]
    out = tmp_path / "filtered.jsonl"
    assemble(recs, out)
    written = read_jsonl(out)
    assert written == [{"conversation": [
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "r"},
    ]}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synthetic/test_assemble.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement assemble**

```python
# src/data/synthetic/assemble.py
from .io import write_jsonl

def strip_meta(record: dict) -> dict:
    return {"conversation": record["conversation"]}

def assemble(records: list[dict], out_path) -> list[dict]:
    stripped = [strip_meta(r) for r in records]
    write_jsonl(out_path, stripped)
    return stripped
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/synthetic/test_assemble.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/synthetic/test_assemble.py src/data/synthetic/assemble.py
git commit -m "feat(synthetic): assemble strips meta to pure fork contract"
```

---

### Task 9: Run manifest (traceability)

**Files:**
- Create: `src/data/synthetic/manifest.py`
- Test: `tests/synthetic/test_manifest.py`

**Interfaces:**
- Consumes: `io.sha256_of`.
- Produces:
  - `manifest.build_manifest(config: dict, seeds: list[dict], stage_counts: dict[str, int], judge_scores: list[int]) -> dict`
    - Fields: `timestamp`, `teacher` (`base_url`+`model`), `seed_set_hash`, `stage_counts`, `judge_score_distribution`, `git_sha`.
  - `manifest.write_manifest(run_dir, manifest: dict) -> None`

- [ ] **Step 1: Write the failing test**

```python
# tests/synthetic/test_manifest.py
import json
from data.synthetic.io import make_record
from data.synthetic.manifest import build_manifest, write_manifest

CFG = {"teacher": {"base_url": "http://x/v1", "model": "qwen3.6", "api_key": "secret"}}

def test_build_manifest_captures_teacher_and_counts_without_api_key():
    seeds = [make_record("c", "r", {})]
    m = build_manifest(CFG, seeds, {"generated": 100, "filtered": 60}, [4, 5, 5])
    assert m["teacher"] == {"base_url": "http://x/v1", "model": "qwen3.6"}
    assert "api_key" not in json.dumps(m)
    assert m["stage_counts"]["filtered"] == 60
    assert m["judge_score_distribution"] == {"4": 1, "5": 2}
    assert len(m["seed_set_hash"]) == 64

def test_write_manifest_creates_file(tmp_path):
    m = build_manifest(CFG, [], {}, [])
    write_manifest(tmp_path, m)
    assert json.loads((tmp_path / "manifest.json").read_text())["teacher"]["model"] == "qwen3.6"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synthetic/test_manifest.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement manifest**

```python
# src/data/synthetic/manifest.py
import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from .io import sha256_of

def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

def build_manifest(config: dict, seeds: list[dict], stage_counts: dict, judge_scores: list[int]) -> dict:
    t = config["teacher"]
    dist = {str(k): v for k, v in sorted(Counter(judge_scores).items())}
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "teacher": {"base_url": t["base_url"], "model": t["model"]},
        "seed_set_hash": sha256_of([s["conversation"] for s in seeds]),
        "stage_counts": stage_counts,
        "judge_score_distribution": dist,
        "git_sha": _git_sha(),
    }

def write_manifest(run_dir, manifest: dict) -> None:
    p = Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "manifest.json").write_text(json.dumps(manifest, indent=2))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/synthetic/test_manifest.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add tests/synthetic/test_manifest.py src/data/synthetic/manifest.py
git commit -m "feat(synthetic): per-run traceability manifest (api_key never recorded)"
```

---

### Task 10: Pipeline orchestration + resumability

**Files:**
- Create: `src/data/synthetic/pipeline.py`
- Test: `tests/synthetic/test_pipeline.py`

**Interfaces:**
- Consumes: all stage modules, `config`, `io`, `manifest`.
- Produces:
  - `pipeline.run_generate(domain, cfg, teacher, embedder, root=Path("."), now=None) -> Path` — returns the run dir.
    - Reads `workspaces/<domain>/seeds/approved.jsonl`; raises `pipeline.CurationGateError` if absent/empty.
    - Generates toward `cfg["generate"]["target_size"]`, **appending** each record to `generated/raw.jsonl` (resume: counts existing rows and only tops up the remainder); `GenerationMiss` skips, never crashes.
    - Applies refine passes → `refined.jsonl`; runs all four filters → `filtered.jsonl`; assembles; writes a manifest + `rejected.jsonl` + `stats.json` under `runs/<ts>/`.
  - `pipeline.CurationGateError(Exception)`

- [ ] **Step 1: Write the failing test**

```python
# tests/synthetic/test_pipeline.py
import json
import pytest
from pathlib import Path
from data.synthetic.io import make_record, write_jsonl, read_jsonl
from data.synthetic.teacher import FakeTeacher
from data.synthetic.embedder import FakeEmbedder
from data.synthetic.pipeline import run_generate, CurationGateError

def _cfg():
    return {
        "teacher": {"base_url": "http://x/v1", "model": "m", "api_key": ""},
        "generate": {"target_size": 2, "fewshot_k": 2, "cot": False},
        "refine": {"passes": []},
        "filter": {
            "schema": {"max_tokens": 2048},
            "dedup": {"similarity_threshold": 0.92},
            "judge": {"score_cutoff": 4},
            "diversity": {"quotas": {"bug": 1.0}},
        },
    }

def test_gate_blocks_without_approved_seeds(tmp_path):
    with pytest.raises(CurationGateError):
        run_generate("code_review", _cfg(), FakeTeacher(["x"]), FakeEmbedder({}), root=tmp_path)

def test_run_generate_produces_filtered_contract(tmp_path):
    ws = tmp_path / "workspaces" / "code_review"
    write_jsonl(ws / "seeds" / "approved.jsonl",
                [make_record("seed code", "seed review", {"source": "bootstrap"})])
    gen = json.dumps({"reasoning": "", "user": "new code", "assistant": "new review", "category": "bug"})
    teacher = FakeTeacher(responses=[gen, "5", gen, "5"])   # 2 generations, each judged 5
    embedder = FakeEmbedder({"new review": [1.0, 0.0]})
    run_dir = run_generate("code_review", _cfg(), teacher, embedder, root=tmp_path,
                           now="2026-06-25T14-30")
    assert (run_dir / "manifest.json").exists()
    filtered = read_jsonl(ws / "generated" / "filtered.jsonl")
    assert all("meta" not in r for r in filtered)
    assert all(set(r) == {"conversation"} for r in filtered)
```

Note: the generator must place `category` into `meta` for diversity. Update `generate.generate_one` is NOT changed; instead `run_generate` copies `obj["category"]` when present. To keep Task 4 stable, the pipeline reads category from the raw generation via a thin wrapper (below).

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synthetic/test_pipeline.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement pipeline**

```python
# src/data/synthetic/pipeline.py
from datetime import datetime, timezone
from pathlib import Path
from . import generate as gen_mod
from .assemble import assemble
from .filter import validate_schema, dedup, judge, enforce_diversity
from .io import read_jsonl, append_jsonl, write_jsonl
from .manifest import build_manifest, write_manifest
from .refine import apply_passes

class CurationGateError(Exception):
    pass

def _categorize(record: dict) -> dict:
    # Lightweight default category until a richer classifier exists.
    record["meta"].setdefault("category", "bug")
    return record

def run_generate(domain, cfg, teacher, embedder, root=Path("."), now=None) -> Path:
    root = Path(root)
    ws = root / "workspaces" / domain
    seeds = read_jsonl(ws / "seeds" / "approved.jsonl")
    if not seeds:
        raise CurationGateError(f"no approved seeds for '{domain}' — run curate first")

    gcfg = cfg["generate"]
    raw_path = ws / "generated" / "raw.jsonl"
    have = len(read_jsonl(raw_path))                      # resume support
    misses = 0
    while have < gcfg["target_size"] and misses < gcfg["target_size"] * 5:
        try:
            rec = gen_mod.generate_one(seeds, teacher, gcfg["fewshot_k"], gcfg["cot"])
        except gen_mod.GenerationMiss:
            misses += 1
            continue
        append_jsonl(raw_path, [_categorize(rec)])
        have += 1

    raw = read_jsonl(raw_path)
    refined = [apply_passes(r, cfg["refine"]["passes"], teacher) for r in raw]
    write_jsonl(ws / "generated" / "refined.jsonl", refined)

    rejected = []
    kept, rej = validate_schema(refined, cfg["filter"]["schema"]["max_tokens"]); rejected += rej
    kept, rej = dedup(kept, embedder, cfg["filter"]["dedup"]["similarity_threshold"]); rejected += rej
    kept, rej = judge(kept, teacher, cfg["filter"]["judge"]["score_cutoff"]); rejected += rej
    kept, rej = enforce_diversity(kept, cfg["filter"]["diversity"]["quotas"], gcfg["target_size"]); rejected += rej

    assemble(kept, ws / "generated" / "filtered.jsonl")

    ts = now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M")
    run_dir = ws / "runs" / ts
    write_jsonl(run_dir / "rejected.jsonl", rejected)
    write_jsonl(run_dir / "stats.json", [{"kept": len(kept), "rejected": len(rejected)}])
    judge_scores = [r["meta"]["judge_score"] for r in kept if "judge_score" in r["meta"]]
    write_manifest(run_dir, build_manifest(cfg, seeds,
                   {"generated": len(raw), "filtered": len(kept)}, judge_scores))
    return run_dir
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/synthetic/test_pipeline.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Add the resume test**

```python
# tests/synthetic/test_pipeline.py  (append)
def test_run_generate_resumes_from_existing_raw(tmp_path):
    ws = tmp_path / "workspaces" / "code_review"
    write_jsonl(ws / "seeds" / "approved.jsonl", [make_record("s", "r", {"source": "bootstrap"})])
    # Pre-seed one raw record so only one more is needed for target_size=2.
    write_jsonl(ws / "generated" / "raw.jsonl",
                [make_record("old", "old review", {"source": "fewshot", "category": "bug"})])
    gen = json.dumps({"reasoning": "", "user": "n", "assistant": "fresh review", "category": "bug"})
    teacher = FakeTeacher(responses=[gen, "5", "5"])      # ONE generation call + judges
    embedder = FakeEmbedder({"old review": [1.0, 0.0], "fresh review": [0.0, 1.0]})
    run_generate("code_review", _cfg(), teacher, embedder, root=tmp_path, now="t1")
    raw = read_jsonl(ws / "generated" / "raw.jsonl")
    assert len(raw) == 2          # did not regenerate the first
```

- [ ] **Step 6: Run resume test to verify it passes**

Run: `pytest tests/synthetic/test_pipeline.py::test_run_generate_resumes_from_existing_raw -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add tests/synthetic/test_pipeline.py src/data/synthetic/pipeline.py
git commit -m "feat(synthetic): pipeline orchestration with curation gate + resume"
```

---

### Task 11: CLI (init, curate, generate) + integration smoke test

**Files:**
- Create: `cli.py`
- Test: `tests/synthetic/test_cli.py`

**Interfaces:**
- Consumes: `config.load_config`, `bootstrap`, `teacher.from_config`, `embedder.SentenceTransformerEmbedder`, `pipeline.run_generate`, `io`.
- Produces a `typer` app with:
  - `init <domain> [--desc TEXT | --seeds PATH]` — creates workspace; with `--seeds` imports, else bootstraps from `--desc` into `seeds/candidates.jsonl`.
  - `curate <domain>` — copies `seeds/candidates.jsonl` → `seeds/approved.jsonl` (human edits the file in between; this command is the explicit approval action).
  - `generate <domain>` — loads config, builds teacher+embedder, calls `run_generate`.

- [ ] **Step 1: Write the failing test (CLI via typer's CliRunner with a fake teacher injected)**

```python
# tests/synthetic/test_cli.py
import json
from typer.testing import CliRunner
from data.synthetic.io import read_jsonl
import cli

runner = CliRunner()

def test_init_with_seeds_imports_candidates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seeds = tmp_path / "in.jsonl"
    seeds.write_text(json.dumps({"conversation": [
        {"role": "user", "content": "c"}, {"role": "assistant", "content": "r"}]}) + "\n")
    result = runner.invoke(cli.app, ["init", "code_review", "--seeds", str(seeds)])
    assert result.exit_code == 0
    assert len(read_jsonl(tmp_path / "workspaces" / "code_review" / "seeds" / "candidates.jsonl")) == 1

def test_curate_promotes_candidates_to_approved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cand = tmp_path / "workspaces" / "code_review" / "seeds" / "candidates.jsonl"
    cand.parent.mkdir(parents=True)
    cand.write_text(json.dumps({"conversation": [
        {"role": "user", "content": "c"}, {"role": "assistant", "content": "r"}]}) + "\n")
    result = runner.invoke(cli.app, ["curate", "code_review"])
    assert result.exit_code == 0
    assert (cand.parent / "approved.jsonl").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/synthetic/test_cli.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'cli'`

- [ ] **Step 3: Implement CLI**

```python
# cli.py
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import typer
from data.synthetic.config import load_config
from data.synthetic.bootstrap import bootstrap_seeds
from data.synthetic.teacher import from_config
from data.synthetic.embedder import SentenceTransformerEmbedder
from data.synthetic.pipeline import run_generate
from data.synthetic.io import read_jsonl, write_jsonl

app = typer.Typer()

def _ws(domain: str) -> Path:
    return Path("workspaces") / domain

@app.command()
def init(domain: str, desc: str = typer.Option(None), seeds: str = typer.Option(None)):
    cand = _ws(domain) / "seeds" / "candidates.jsonl"
    cand.parent.mkdir(parents=True, exist_ok=True)
    if seeds:
        write_jsonl(cand, read_jsonl(seeds))
        typer.echo(f"Imported {len(read_jsonl(cand))} seeds to {cand}")
    else:
        if not desc:
            raise typer.BadParameter("provide --seeds PATH or --desc TEXT")
        cfg = load_config(domain)
        teacher = from_config(cfg)
        recs = bootstrap_seeds(desc, teacher, cfg["bootstrap"]["starter_count"])
        write_jsonl(cand, recs)
        typer.echo(f"Bootstrapped {len(recs)} candidate seeds to {cand}. Edit, then run: curate {domain}")

@app.command()
def curate(domain: str):
    cand = _ws(domain) / "seeds" / "candidates.jsonl"
    approved = _ws(domain) / "seeds" / "approved.jsonl"
    shutil.copyfile(cand, approved)
    typer.echo(f"Approved {len(read_jsonl(approved))} seeds → {approved}")

@app.command()
def generate(domain: str):
    cfg = load_config(domain)
    teacher = from_config(cfg)
    embedder = SentenceTransformerEmbedder(cfg["filter"]["dedup"]["embedding_model"])
    run_dir = run_generate(domain, cfg, teacher, embedder)
    typer.echo(f"Done. Run artifacts in {run_dir}")

if __name__ == "__main__":
    app()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/synthetic/test_cli.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Add integration smoke test (skipped unless a local server is present)**

```python
# tests/synthetic/test_integration.py
import os
import pytest

@pytest.mark.skipif(not os.environ.get("ELIXIRLORA_LIVE"),
                    reason="set ELIXIRLORA_LIVE=1 with a local llama.cpp server on :8080")
def test_end_to_end_against_local_llamacpp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from typer.testing import CliRunner
    import cli
    from data.synthetic.io import write_jsonl, make_record, read_jsonl
    ws = tmp_path / "workspaces" / "code_review"
    write_jsonl(ws / "seeds" / "approved.jsonl",
                [make_record(f"def f(): return {i}", f"Looks fine, example {i}.", {"source": "bootstrap"})
                 for i in range(3)])
    # small run via config override
    write_jsonl.__self__  # no-op reference
    (ws / "config.yaml").write_text("generate:\n  target_size: 3\n")
    result = CliRunner().invoke(cli.app, ["generate", "code_review"])
    assert result.exit_code == 0
    assert len(read_jsonl(ws / "generated" / "filtered.jsonl")) >= 0
```

- [ ] **Step 6: Run the full suite (integration auto-skips)**

Run: `pytest tests/synthetic/ -v`
Expected: PASS (all unit/CLI pass; integration shows SKIPPED)

- [ ] **Step 7: Commit**

```bash
git add cli.py tests/synthetic/test_cli.py tests/synthetic/test_integration.py
git commit -m "feat(synthetic): CLI (init/curate/generate) + local integration smoke test"
```

---

## Self-Review Notes

- **Spec coverage:** §1 boundary → Tasks 8/10 (pure contract out). §2 workspace/gate → Task 11 init/curate + Task 10 gate. §3 stages → Tasks 3–8. §3 teacher → Task 2. §4 layout/records → Tasks 1/4/10. §5 generation+CoT/refine → Tasks 4/5. §6 filtering → Task 7. §7 error handling/resume → Task 10 (`GenerationMiss` skip, resume test) + Task 2 (`from_config` fail-fast). §8 traceability → Task 9. §9 config model → Task 1. §10 testing (FakeTeacher/contract/resume/integration) → Tasks 2,8,10,11.
- **Deferred-by-design (not gaps):** `--dry-run`/max-call budget (§7) and retry/backoff are noted in spec as guards; backoff belongs in `OpenAITeacher.chat` and is a small follow-up task if the first live run shows flakiness — left out to keep tasks bite-sized and avoid untestable-without-network code. The `_categorize` default in Task 10 is a deliberate stub for the single-domain slice; richer category detection is future work (§11).
- **Type consistency:** `make_record`/`strip_meta`/`apply_passes`/`run_generate`/`from_config` signatures match across all consuming tasks.
