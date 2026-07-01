# Rename: elixirlora → elixirtune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace every user-visible and code reference to "ElixirLoRA" / "elixirlora" with "ElixirTune" / "elixirtune" across the entire repository.

**Architecture:** Pure search-and-replace. No logic changes. The internal module structure (`src.*`, `tui.*`, `commands.*`) is unchanged — nothing is imported by the project name, so no import rewiring is needed.

**Tech Stack:** Python, Textual TUI, Typer CLI.

## Global Constraints

- Internal import paths (`src.*`, `tui.*`, `commands.*`) must not change.
- All existing tests must pass after every task.
- Directory rename and GitHub repo rename are manual steps — documented at the end, not scripted.

---

### Task 1: Rename Python source files

**Files:**
- Modify: `tui/app.py` (class name + TITLE string)
- Modify: `cli.py` (import + docstring)
- Modify: `requirements.txt` (comment)
- Modify: `tests/tui/test_app.py` (all 5 `ElixirLoRAApp` references)
- Modify: `tests/test_code_review/test_integration.py` (any references)
- Modify: `examples/code_review/setup.py` (any references)

**Interfaces:**
- Produces: `ElixirTuneApp` in `tui/app.py` — everything that imported `ElixirLoRAApp` must use this name.

- [ ] **Step 1: Update `tui/app.py`**

Change line 31–32:
```python
# Before
class ElixirLoRAApp(App):
    TITLE = "ElixirLoRA"

# After
class ElixirTuneApp(App):
    TITLE = "ElixirTune"
```

- [ ] **Step 2: Update `cli.py`**

Change lines 33–36:
```python
# Before
@app.command()
def tui(domain: str = typer.Option(None, help="Domain to pre-select on launch")):
    """Launch the ElixirLoRA TUI."""
    from tui.app import ElixirLoRAApp
    ElixirLoRAApp(initial_domain=domain).run()

# After
@app.command()
def tui(domain: str = typer.Option(None, help="Domain to pre-select on launch")):
    """Launch the ElixirTune TUI."""
    from tui.app import ElixirTuneApp
    ElixirTuneApp(initial_domain=domain).run()
```

- [ ] **Step 3: Update `requirements.txt` comment**

Change line 31:
```
# Before
# ElixirLoRA synthetic data pipeline

# After
# ElixirTune synthetic data pipeline
```

- [ ] **Step 4: Update `tests/tui/test_app.py`**

Replace all five occurrences of `ElixirLoRAApp` → `ElixirTuneApp`:
```python
# Before (line 3)
from tui.app import ElixirLoRAApp

# After
from tui.app import ElixirTuneApp
```

Then replace every `ElixirLoRAApp(` → `ElixirTuneApp(` in the file (appears on lines 9, 16, 24, 31, 40).

- [ ] **Step 5: Update `tests/test_code_review/test_integration.py` and `examples/code_review/setup.py`**

Run to see exact lines:
```bash
grep -n "ElixirLoRAApp\|ElixirLoRA\|elixirlora" tests/test_code_review/test_integration.py examples/code_review/setup.py
```

For each occurrence found, replace `ElixirLoRA` → `ElixirTune` and `elixirlora` → `elixirtune`.

- [ ] **Step 6: Verify no remaining old references in Python files**

```bash
grep -r "ElixirLoRAApp\|ElixirLoRA\|elixirlora" . --include="*.py" | grep -v .venv | grep -v __pycache__
```
Expected: no output.

- [ ] **Step 7: Run tests**

```bash
.venv/bin/python -m pytest tests/ -x -q 2>&1 | tail -20
```
Expected: all tests pass. Fix any failures before committing.

- [ ] **Step 8: Commit**

```bash
git add tui/app.py cli.py requirements.txt tests/tui/test_app.py tests/test_code_review/test_integration.py examples/code_review/setup.py
git commit -m "feat: rename ElixirLoRAApp → ElixirTuneApp and update all code references"
```

---

### Task 2: Update README and example docs

**Files:**
- Modify: `README.md`
- Modify: `examples/code_review/README.md`

**Interfaces:**
- Consumes: nothing
- Produces: updated docs — no code depends on these

- [ ] **Step 1: Update `README.md` — title and description**

Line 1: `# ElixirLoRA` → `# ElixirTune`

Line 3–5 (description paragraph), change from:
```
Fine-tuning an LLM involves many moving parts — data preparation, training runs, evaluation, adapter fusion, and deployment. ElixirLoRA is a LoRA fine-tuning workbench for Apple Silicon that organizes the full pipeline into a single managed workspace per domain, driven by a guided TUI or CLI.
```
to:
```
Fine-tuning a model involves many moving parts — data preparation, training runs, evaluation, adapter fusion, and deployment. ElixirTune is a fine-tuning workbench for Apple Silicon that organizes the full pipeline into a single managed workspace per domain, driven by a guided TUI or CLI.
```

- [ ] **Step 2: Replace remaining `ElixirLoRA` references in `README.md`**

```bash
grep -n "ElixirLoRA\|elixirlora" README.md
```

For each occurrence: replace `ElixirLoRA` → `ElixirTune` and `elixirlora` → `elixirtune`.

- [ ] **Step 3: Update `examples/code_review/README.md`**

```bash
grep -n "ElixirLoRA\|elixirlora" examples/code_review/README.md
```
Replace each occurrence found.

- [ ] **Step 4: Verify**

```bash
grep -ri "elixirlora" README.md examples/code_review/README.md
```
Expected: no output.

- [ ] **Step 5: Commit**

```bash
git add -f README.md examples/code_review/README.md
git commit -m "docs: rename ElixirLoRA → ElixirTune in README and example docs"
```

---

### Task 3: Manual post-commit steps (user action required)

These steps are NOT scripted — they must be done by the user after the code commits above are complete.

- [ ] **Step 1: Rename local directory**

```bash
cd /Users/jinzuo/projects
mv elixirlora elixirtune
cd elixirtune
```

- [ ] **Step 2: Rename the GitHub repository**

```bash
gh repo rename elixirtune
```
GitHub automatically sets up a redirect from the old URL. Existing clones do not need their remote URL changed for read access, but to update the local remote:
```bash
git remote set-url origin https://github.com/<your-username>/elixirtune.git
```

- [ ] **Step 3: Verify the TUI still launches with the new title**

```bash
.venv/bin/python cli.py tui --help
```
Expected: output shows `Launch the ElixirTune TUI.`
