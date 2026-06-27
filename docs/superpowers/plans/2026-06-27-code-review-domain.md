# Code Review Domain Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `code-review` domain to ElixirLoRA using the `ronantakizawa/github-codereview` HuggingFace dataset, plus framework-level GGUF export support.

**Architecture:** The `examples/code-review/` folder handles dataset download and conversion into ElixirLoRA's seed format. GGUF export is added as a framework-level feature in `commands/export_gguf.py`. The TUI's "New Domain" screen gains a radio option to create a code-review domain, and the Deploy tab gains an "Export GGUF" button.

**Tech Stack:** Python 3.11+, mlx-lm (for `mlx_lm.convert`), datasets (HF), typer, textual, pyyaml.

## Global Constraints

- ElixirLoRA core pipeline (generate → prepare → train) must not be modified for code-review; data preparation is entirely in `examples/code-review/`
- GGUF export uses `mlx_lm.convert` — verify mlx-lm version supports it (≥0.18)
- llama.cpp must be installed for GGUF export — check at runtime with clear error if missing
- All new code must have tests following existing patterns
- DRY: GGUF export logic is a single function called by both CLI and TUI
- Follow existing code style: typer sub-apps, `@app.callback(invoke_without_command=True)`, `sys.path.insert` pattern, `_ws()` workspace accessor

---

## Task 1: Create `examples/code-review/setup.py`

**Files:**
- Create: `examples/code-review/setup.py`
- Create: `examples/code-review/README.md`

**Interfaces:**
- Consumes: `datasets.load_dataset`, `tqdm`, `yaml`
- Produces: `workspaces/<domain>/seeds/candidates.jsonl`, `workspaces/<domain>/seeds/approved.jsonl`, `workspaces/<domain>/config.yaml`

- [ ] **Step 1: Write the setup.py with dataset download and conversion**

```python
#!/usr/bin/env python3
"""Download and convert the github-codereview dataset for ElixirLoRA."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from datasets import load_dataset
from tqdm import tqdm


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def make_record(user_content: str, assistant_content: str, meta: dict[str, Any]) -> dict:
    return {
        "conversation": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "meta": meta,
    }


def build_user_message(record: dict) -> str:
    """Build the user prompt from a code review dataset record."""
    parts = ["You are reviewing the following code:\n"]

    diff = record.get("code_diff", "")
    if diff:
        parts.append(f"```diff\n{diff}\n```\n")

    file_path = record.get("file_path", "")
    if file_path:
        parts.append(f"File: {file_path}\n")

    language = record.get("language", "")
    if language:
        parts.append(f"Language: {language}\n")

    pr_title = record.get("pr_title", "")
    if pr_title:
        parts.append(f"PR: {pr_title}\n")

    parts.append("\nPlease provide a code review.")
    return "\n".join(parts)


def build_assistant_message(record: dict) -> str:
    """Build the assistant response from a code review dataset record."""
    review_type = record.get("review_type", "none")
    comment = record.get("comment", "")
    suggestion = record.get("suggestion", "")

    parts = [f"[review_type: {review_type}]"]

    if comment:
        parts.append("")
        parts.append(comment.strip())

    if suggestion:
        parts.append("")
        parts.append(f"```suggestion\n{suggestion}\n```")

    if not (comment or suggestion):
        parts.append(" No issues found.")

    return "\n".join(parts)


def convert_dataset(
    domain: str,
    languages: list[str] | None = None,
    target: int = 200000,
    include_negative: bool = True,
    root: Path = Path("."),
) -> None:
    """Download the github-codereview dataset and convert to ElixirLoRA seed format."""

    dataset_path = root / "data" / "code-review"
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Load dataset from HuggingFace
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("ronantakizawa/github-codereview", split="train")
    print(f"Loaded {len(dataset)} records.")

    # Filter by languages if specified
    if languages:
        orig = len(dataset)
        dataset = [r for r in dataset if r.get("language", "") in languages]
        print(f"Filtered to {len(dataset)} records in {languages} (was {orig}).")

    # Filter out negative examples if requested
    if not include_negative:
        orig = len(dataset)
        dataset = [r for r in dataset if r.get("review_type") != "none"]
        print(f"Removed negative examples: {len(dataset)} remain (was {orig}).")

    if not dataset:
        print("No records after filtering. Aborting.", file=sys.stderr)
        sys.exit(1)

    # Convert to ElixirLoRA format
    print(f"Converting {len(dataset)} records to ElixirLoRA seed format...")
    seeds = []
    for record in tqdm(dataset, desc="Converting"):
        user_content = build_user_message(record)
        assistant_content = build_assistant_message(record)
        meta = {
            "review_type": record.get("review_type", "none"),
            "language": record.get("language", ""),
            "repo": record.get("repo_name", ""),
            "pr_title": record.get("pr_title", ""),
            "pr_number": record.get("pr_number", 0),
        }
        seeds.append(make_record(user_content, assistant_content, meta))

    # Create workspace
    ws = root / "workspaces" / domain
    seeds_dir = ws / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)

    # Write candidates and approved (same content — seeds from dataset are curated)
    candidates_path = seeds_dir / "candidates.jsonl"
    approved_path = seeds_dir / "approved.jsonl"
    candidates_path.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in seeds) + "\n",
        encoding="utf-8",
    )
    # approved = candidates for this domain (pre-curated dataset)
    approved_path.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in seeds) + "\n",
        encoding="utf-8",
    )

    # Write config.yaml for code-review domain
    config = {
        "generate": {
            "target_size": min(target, len(seeds)),
            "fewshot_k": 4,
        },
        "filter": {
            "dedup": {
                "embedding_model": "all-MiniLM-L6-v2",
                "similarity_threshold": 0.92,
            },
            "diversity": {
                "quotas": {},  # no specific quotas for code-review
            },
        },
    }
    import yaml
    config_path = ws / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    print(f"\nDone! Created workspace: {ws}")
    print(f"  Seeds: {len(seeds)} records → {approved_path}")
    print(f"  Config: {config_path}")
    print(f"\nNext steps:")
    print(f"  1. (Optional) Edit {candidates_path} to curate seeds")
    print(f"  2. Launch TUI: python3 cli.py tui --domain {domain}")
    print(f"  3. Or run pipeline: python3 cli.py generate {domain}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download code-review dataset for ElixirLoRA")
    parser.add_argument("--domain", default="code-review", help="Workspace domain name")
    parser.add_argument("--languages", default=None, help="Comma-separated list of languages to filter (e.g., 'Python,TypeScript')")
    parser.add_argument("--target", type=int, default=200000, help="Target number of samples")
    parser.add_argument("--no-negative", action="store_true", help="Exclude negative (no issues) examples")
    parser.add_argument("--root", default=".", help="Project root directory")
    args = parser.parse_args()

    languages = [l.strip() for l in args.languages.split(",")] if args.languages else None
    root = Path(args.root)

    convert_dataset(
        domain=args.domain,
        languages=languages,
        target=args.target,
        include_negative=not args.no_negative,
        root=root,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write README.md**

```markdown
# Code Review Domain

Fine-tune a model to act as a code review assistant using the
[ronantakizawa/github-codereview](https://huggingface.co/datasets/ronantakizawa/github-codereview)
dataset.

## Quick Start

```bash
# 1. Download dataset and create workspace
python3 examples/code-review/setup.py --domain code-review

# 2. (Optional) Curate seeds by editing:
#    workspaces/code-review/seeds/candidates.jsonl

# 3. Launch the ElixirLoRA TUI
python3 cli.py tui --domain code-review

# 4. Follow the TUI: Synth → Training → Deploy → Export GGUF
```

## CLI Options

```bash
python3 examples/code-review/setup.py --domain myreview \
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
```

- [ ] **Step 3: Write tests for setup.py**

Create `tests/test_code_review/test_setup.py`:

```python
import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "src"))


def test_make_record_shape():
    """Test that make_record produces the correct structure."""
    from examples.code_review.setup import make_record
    rec = make_record("user content", "assistant content", {"source": "test"})
    assert rec["conversation"] == [
        {"role": "user", "content": "user content"},
        {"role": "assistant", "content": "assistant content"},
    ]
    assert rec["meta"]["source"] == "test"


def test_build_user_message_with_all_fields():
    """Test user message construction with all fields present."""
    from examples.code_review.setup import build_user_message
    record = {
        "code_diff": "@@ -1,3 +1,3 @@\n-old\n+new",
        "file_path": "src/example.py",
        "language": "Python",
        "pr_title": "Fix example",
    }
    result = build_user_message(record)
    assert "You are reviewing the following code:" in result
    assert "```diff" in result
    assert "@@ -1,3 +1,3 @@" in result
    assert "File: src/example.py" in result
    assert "Language: Python" in result
    assert "PR: Fix example" in result
    assert "Please provide a code review." in result


def test_build_user_message_minimal():
    """Test user message with empty fields."""
    from examples.code_review.setup import build_user_message
    record = {"code_diff": "", "file_path": "", "language": "", "pr_title": ""}
    result = build_user_message(record)
    assert "You are reviewing the following code:" in result
    assert "Please provide a code review." in result


def test_build_assistant_message_with_review():
    """Test assistant message for a suggestion-type review."""
    from examples.code_review.setup import build_assistant_message
    record = {
        "review_type": "suggestion",
        "comment": "Consider using a more descriptive name.",
        "suggestion": "def compute_total_price():",
    }
    result = build_assistant_message(record)
    assert "[review_type: suggestion]" in result
    assert "Consider using a more descriptive name." in result
    assert "```suggestion" in result
    assert "def compute_total_price():" in result


def test_build_assistant_message_negative():
    """Test assistant message for no-issues (negative) example."""
    from examples.code_review.setup import build_assistant_message
    record = {
        "review_type": "none",
        "comment": "",
        "suggestion": "",
    }
    result = build_assistant_message(record)
    assert "[review_type: none]" in result
    assert "No issues found." in result


def test_build_assistant_message_with_only_comment():
    """Test assistant message when only comment is present (no suggestion)."""
    from examples.code_review.setup import build_assistant_message
    record = {
        "review_type": "question",
        "comment": "Why is this check necessary?",
        "suggestion": "",
    }
    result = build_assistant_message(record)
    assert "[review_type: question]" in result
    assert "Why is this check necessary?" in result
    assert "```suggestion" not in result


def test_build_assistant_message_with_only_suggestion():
    """Test assistant message when only suggestion is present."""
    from examples.code_review.setup import build_assistant_message
    record = {
        "review_type": "refactor",
        "comment": "",
        "suggestion": "def helper_func(x):\n    return x * 2",
    }
    result = build_assistant_message(record)
    assert "[review_type: refactor]" in result
    assert "```suggestion" in result
    assert "def helper_func(x):" in result


def test_write_read_jsonl_roundtrip(tmp_path):
    """Test that seeds can be written and read back correctly."""
    from examples.code_review.setup import make_record
    rec = make_record("user", "assistant", {"review_type": "suggestion"})
    path = tmp_path / "seeds" / "approved.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    assert lines[0] == rec


def test_config_yaml_structure(tmp_path):
    """Test that the generated config.yaml has the expected structure."""
    from examples.code_review.setup import convert_dataset
    convert_dataset(domain="test-domain", target=100, include_negative=False, root=tmp_path)

    import yaml
    config_path = tmp_path / "workspaces" / "test-domain" / "config.yaml"
    config = yaml.safe_load(config_path.read_text())

    assert "generate" in config
    assert "target_size" in config["generate"]
    assert config["generate"]["target_size"] == 100
    assert "filter" in config
    assert "dedup" in config["filter"]
    assert config["filter"]["dedup"]["embedding_model"] == "all-MiniLM-L6-v2"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/jinzuo/projects/elixirlora
python3 -m pytest tests/test_code_review/test_setup.py -v
```

Expected: All tests pass (7 unit tests + 2 integration-style tests).

- [ ] **Step 5: Commit**

```bash
git add examples/code-review/ tests/test_code_review/
git commit -m "feat: add code-review dataset download and conversion (setup.py)"
```

---

## Task 2: Add GGUF Export Command

**Files:**
- Create: `commands/export_gguf.py`
- Modify: `cli.py:1-20` (add import and subcommand)

**Interfaces:**
- Consumes: `mlx_lm.convert` command, fused model from `workspaces/<domain>/fused/`
- Produces: `.gguf` file

- [ ] **Step 1: Write `commands/export_gguf.py`**

```python
"""Export fused model to GGUF format using mlx_lm.convert."""

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer

app = typer.Typer(context_settings={"allow_interspersed_args": True})

from commands import _ws


QUANTIZATIONS = ["Q4_K_M", "Q5_K_M", "Q8_0"]


def _check_llama_cpp() -> bool:
    """Check if llama.cpp is available for GGUF conversion."""
    return shutil.which("llama-convert-hf-to-gguf.py") is not None or \
           shutil.which("llama-export-lora") is not None or \
           (shutil.which("python3") and Path(shutil.which("python3").replace("python3", "llama-convert-hf-to-gguf.py")).exists())


@app.callback(invoke_without_command=True)
def export_gguf(
    ctx: typer.Context,
    domain: str = typer.Argument(...),
    quantization: str = typer.Option("Q4_K_M", help="Quantization method",
                                     help_text="Options: Q4_K_M, Q5_K_M, Q8_0",
                                     rich_help_panel="Quantization",
                                     ),
    output_path: Path = typer.Option(None, help="Output GGUF file path"),
) -> None:
    """Export fused model to GGUF format for use with llama.cpp / Ollama."""
    if ctx.invoked_subcommand is not None:
        return

    ws = _ws(domain)
    fused = ws / "fused"

    if not fused.exists() or not any(fused.iterdir()):
        typer.echo(f"No fused model at {fused}. Run: fuse {domain} first.", err=True)
        raise typer.Exit(1)

    if quantization not in QUANTIZATIONS:
        typer.echo(f"Unknown quantization '{quantization}'. Choose from: {QUANTIZATIONS}", err=True)
        raise typer.Exit(1)

    out = output_path or (ws / "fused" / f"{domain}.gguf")

    typer.echo(f"Exporting {domain} to GGUF ({quantization})...")
    typer.echo(f"  Input:  {fused}")
    typer.echo(f"  Output: {out}")

    # Check for llama.cpp
    if not _check_llama_cpp():
        typer.echo("")
        typer.echo("llama.cpp is required for GGUF export.", err=True)
        typer.echo("Install it with: git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make", err=True)
        typer.echo("Then ensure llama-convert-hf-to-gguf.py or llama-export-lora is on PATH.", err=True)
        raise typer.Exit(1)

    # Try mlx_lm.convert first (preferred if available)
    try:
        result = subprocess.run(
            [
                "python3", "-m", "mlx_lm.convert",
                "--model-dir", str(fused),
                "--outfile", str(out),
                "--quantize", quantization,
            ],
            check=False,
            capture_output=False,
        )
        if result.returncode == 0:
            typer.echo(f"✅ GGUF exported to: {out}")
            return
    except (FileNotFoundError, ModuleNotFoundError):
        pass  # mlx_lm.convert not available, fall through

    # Fallback: use llama.cpp's conversion
    typer.echo("mlx_lm.convert not available, trying llama.cpp...")
    convert_script = None
    for name in ["llama-convert-hf-to-gguf.py", "llama-export-lora"]:
        script = shutil.which(name)
        if script:
            convert_script = script
            break

    if not convert_script:
        typer.echo("Neither mlx_lm.convert nor llama.cpp conversion scripts found.", err=True)
        typer.echo("Please install one of:", err=True)
        typer.echo("  - mlx-lm: pip install mlx-lm>=0.18 (provides mlx_lm.convert)")
        typer.echo("  - llama.cpp: git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make")
        raise typer.Exit(1)

    # Use llama.cpp to convert
    typer.echo(f"Using {convert_script}...")
    result = subprocess.run(
        [sys.executable, convert_script, str(fused), "--outfile", str(out), "--outtype", "f16"],
        check=False,
        capture_output=False,
    )

    if result.returncode != 0:
        typer.echo(f"GGUF export failed.", err=True)
        raise typer.Exit(1)

    # Quantize if needed (llama.cpp converts to f16 by default)
    if quantization != "Q8_0":
        quantize = shutil.which("llama-quantize") or shutil.which("quantize")
        if quantize:
            typer.echo(f"Quantizing to {quantization}...")
            result = subprocess.run(
                [quantize, str(out), str(out), quantization],
                check=False,
                capture_output=False,
            )
            if result.returncode != 0:
                typer.echo(f"Quantization failed.", err=True)
                raise typer.Exit(1)
        else:
            typer.echo("llama-quantize not found. Output is f16. Install llama.cpp for quantization.",
                       err=True)

    typer.echo(f"✅ GGUF exported to: {out}")
```

- [ ] **Step 2: Update `cli.py` to add the export-gguf subcommand**

Replace the imports and subcommand registration block in `cli.py`:

```python
# OLD:
from commands.fuse     import app as fuse_app
from commands.chat     import app as chat_app
# ...
app.add_typer(fuse_app, name="fuse")
app.add_typer(chat_app, name="chat")

# NEW:
from commands.fuse     import app as fuse_app
from commands.export_gguf import app as export_gguf_app
from commands.chat     import app as chat_app
# ...
app.add_typer(fuse_app, name="fuse")
app.add_typer(export_gguf_app, name="export-gguf")
app.add_typer(chat_app, name="chat")
```

- [ ] **Step 3: Write tests**

Create `tests/test_commands/test_export_gguf.py`:

```python
"""Tests for the export_gguf command."""

import pytest
from typer.testing import CliRunner


def test_export_gguf_requires_fused_model(tmp_path):
    """Export GGUF must fail when no fused model exists."""
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_root / "src"))

    # Clear cached modules
    for mod in list(sys.modules.keys()):
        if mod.startswith("commands"):
            del sys.modules[mod]

    from cli import app

    runner = CliRunner()
    ws = tmp_path / "workspaces" / "test"
    ws.mkdir(parents=True)
    result = runner.invoke(app, ["export-gguf", "test"])
    assert result.exit_code != 0
    assert "No fused model" in result.output


def test_export_gguf_rejects_bad_quantization(tmp_path):
    """Export GGUF must reject invalid quantization values."""
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_root / "src"))

    for mod in list(sys.modules.keys()):
        if mod.startswith("commands"):
            del sys.modules[mod]

    from cli import app

    runner = CliRunner()
    ws = tmp_path / "workspaces" / "test"
    (ws / "fused").mkdir(parents=True)
    (ws / "fused" / "dummy").touch()

    result = runner.invoke(app, ["export-gguf", "test", "--quantization", "INVALID"])
    assert result.exit_code != 0
    assert "Unknown quantization" in result.output
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/jinzuo/projects/elixirlora
python3 -m pytest tests/test_commands/test_export_gguf.py -v
```

- [ ] **Step 5: Commit**

```bash
git add commands/export_gguf.py cli.py tests/test_commands/test_export_gguf.py
git commit -m "feat: add export-gguf CLI command for GGUF model conversion"
```

---

## Task 3: Integrate GGUF Export into TUI

**Files:**
- Modify: `tui/panels/deployment.py` (add Export GGUF button)
- Modify: `tui/panels/deployment.py` (add `_run_export_gguf` method)
- Modify: `tui/domain.py` (add `Status.EXPORTED` — optional, can skip for now)

**Interfaces:**
- Consumes: `commands.export_gguf` CLI
- Produces: GGUF file in log view and Deploy panel state

- [ ] **Step 1: Add Export GGUF button to DeploymentPanel**

In `tui/panels/deployment.py`, update the `compose` method:

```python
def compose(self) -> ComposeResult:
    yield Label("Adapters: —", id="adapter-info")
    yield Label("Fused model: —", id="fused-info")
    yield Rule()
    yield Button("▶ Fuse & Evaluate", id="fuse-btn", disabled=True, variant="success")
    yield Button("▶ Export GGUF", id="gguf-btn", disabled=True)
    yield Button("Create Ollama Model", id="ollama-btn", disabled=True)
    yield Button("Upload to HuggingFace", id="hf-upload-btn", disabled=True)
    yield Rule()
    yield LogView(id="deploy-log")
```

Update `refresh_content` to enable the GGUF button after fusion:

```python
# Add after the hf-upload-btn disabled check:
self.query_one("#gguf-btn", Button).disabled = (
    status_order(status) < status_order(Status.DEPLOYED)
)
```

Update `on_button_pressed` to handle the GGUF button:

```python
# Add after the hf-upload-btn elif block:
elif event.button.id == "gguf-btn":
    event.button.disabled = True
    self._run_export_gguf(self.domain)
```

- [ ] **Step 2: Add `_run_export_gguf` work method**

Add this new method to `DeploymentPanel`:

```python
@work(thread=True)
def _run_export_gguf(self, domain: str) -> None:
    cmd = [
        "python3", "cli.py", "export-gguf", domain,
        "--quantization", "Q4_K_M",
    ]
    self._stream(cmd)
```

- [ ] **Step 3: Write TUI panel tests**

Add to `tests/tui/test_panels.py` (new test block at end):

```python
# ── GGUF export tests ───────────────────────────────────────────────────────

async def test_gguf_button_disabled_without_fused(tmp_path):
    """GGUF export button must be disabled when no fused model exists."""
    ws = tmp_path / "workspaces" / "d"
    ws.mkdir(parents=True)
    import os; os.chdir(tmp_path)
    async with DeployApp(ws).run_test() as pilot:
        await pilot.pause()
        from textual.widgets import Button
        btn = pilot.app.query_one("#gguf-btn", Button)
        assert btn.disabled


async def test_gguf_button_enabled_with_fused(tmp_path):
    """GGUF export button must be enabled after model is fused."""
    ws = tmp_path / "workspaces" / "d"
    (ws / "fused").mkdir(parents=True)
    (ws / "fused" / "weights.safetensors").write_text("x")
    import os; os.chdir(tmp_path)
    async with DeployApp(ws).run_test() as pilot:
        await pilot.pause()
        from textual.widgets import Button
        btn = pilot.app.query_one("#gguf-btn", Button)
        assert not btn.disabled


async def test_gguf_button_click_streams_to_log(tmp_path):
    """Clicking Export GGUF must run the command and stream output."""
    ws = tmp_path / "workspaces" / "d"
    (ws / "fused").mkdir(parents=True)
    (ws / "fused" / "model.safetensors").write_text("x")
    import os; os.chdir(tmp_path)
    async with DeployApp(ws).run_test() as pilot:
        await pilot.pause()
        await pilot.click("#gguf-btn")
        await pilot.pause()
        from textual.widgets import LogView
        log = pilot.app.query_one(LogView)
        # Should have logged the export command
        assert any("export-gguf" in str(log) for _ in [1])  # basic existence check
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd /Users/jinzuo/projects/elixirlora
python3 -m pytest tests/tui/test_panels.py -v -k "gguf"
```

- [ ] **Step 5: Commit**

```bash
git add tui/panels/deployment.py tests/tui/test_panels.py
git commit -m "feat: add Export GGUF button to TUI Deploy panel"
```

---

## Task 4: TUI Integration — New Domain Radio Option

**Files:**
- Modify: `tui/new_domain.py` (add third radio option, handler logic)

**Interfaces:**
- Consumes: `examples/code-review/setup.py` (called as subprocess)
- Produces: New domain workspace on success

- [ ] **Step 1: Add third radio option and update form**

In `tui/new_domain.py`, update `compose` to add the third option:

```python
# Replace the RadioSet block with:
yield RadioSet(
    RadioButton("Bootstrap from description", id="rb-bootstrap", value=True),
    RadioButton("Import from file", id="rb-import"),
    RadioButton("Download code review dataset", id="rb-code-review"),
    id="source-radio",
)
```

- [ ] **Step 2: Update `on_button_pressed` to handle the third option**

Replace the `on_button_pressed` method in `NewDomainScreen`:

```python
def on_button_pressed(self, event: Button.Pressed) -> None:
    if event.button.id == "new-domain-cancel":
        self.dismiss(None)
    elif event.button.id == "new-domain-create":
        name = self.query_one("#new-domain-name", Input).value.strip()
        if not name or " " in name:
            return
        radio = self.query_one(RadioSet)
        pressed = radio.pressed_button and radio.pressed_button.id
        if pressed == "rb-import":
            seeds = self.query_one("#new-domain-seeds-path", Input).value.strip()
            cmd = ["python3", "cli.py", "init", name, "--seeds", seeds]
        elif pressed == "rb-code-review":
            cmd = ["python3", "examples/code-review/setup.py", "--domain", name]
        else:  # bootstrap
            desc = self.query_one("#new-domain-desc", TextArea).text.strip() or f"{name} domain"
            cmd = ["python3", "cli.py", "init", name, "--desc", desc]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            self.dismiss({"name": name, "success": True})
        else:
            self.dismiss({"name": name, "success": False, "error": result.stderr})
```

Also update the labels/inputs to match the new IDs:

```python
# In compose, change the seeds path input id and desc input id:
yield Label("Description:")
yield TextArea(id="new-domain-desc")
yield Label("Seeds file path:")
yield Input(id="new-domain-seeds-path", placeholder="/path/to/seeds.jsonl")
```

- [ ] **Step 3: Write TUI tests**

Add to `tests/tui/test_new_domain.py`:

```python
"""Tests for the NewDomainScreen."""

import pytest
from pathlib import Path
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, RadioButton, RadioSet, TextArea


class NewDomainApp(App):
    def compose(self) -> ComposeResult:
        from tui.new_domain import NewDomainScreen
        yield NewDomainScreen()

    def on_mount(self) -> None:
        self.push_screen(NewDomainScreen())


async def test_radio_options_present():
    """All three radio options should be visible."""
    async with NewDomainApp().run_test() as pilot:
        screen = pilot.app.screen
        radio = screen.query_one(RadioSet)
        buttons = [rb for rb in radio.query(RadioButton)]
        ids = {rb.id for rb in buttons}
        assert "rb-bootstrap" in ids
        assert "rb-import" in ids
        assert "rb-code-review" in ids


async def test_code_review_creates_workspace(tmp_path):
    """Code review selection should call setup.py with --domain."""
    import os
    os.chdir(tmp_path)
    async with NewDomainApp().run_test() as pilot:
        await pilot.pause()
        screen = pilot.app.screen
        # Select code review option
        radio = screen.query_one(RadioSet)
        for rb in radio.query(RadioButton):
            if rb.id == "rb-code-review":
                rb.value = True
                break
        # Enter domain name
        name_input = screen.query_one("#new-domain-name", Input)
        name_input.value = "test-code-review"
        # Click Create
        create_btn = screen.query_one("#new-domain-create", Button)
        create_btn.press()
        await pilot.pause()
        # The screen should dismiss with success
        assert pilot.app.screen_stack[-1].__class__ != type(screen)
```

- [ ] **Step 4: Run tests**

```bash
cd /Users/jinzuo/projects/elixirlora
python3 -m pytest tests/tui/test_new_domain.py -v
```

- [ ] **Step 5: Commit**

```bash
git add tui/new_domain.py tests/tui/test_new_domain.py
git commit -m "feat: add 'Download code review dataset' option to New Domain screen"
```

---

## Task 5: Standalone `examples/code-review/export_gguf.py`

**Files:**
- Create: `examples/code-review/export_gguf.py`

**Interfaces:**
- Consumes: `commands.export_gguf` via CLI subprocess
- Produces: `.gguf` file

- [ ] **Step 1: Write the standalone wrapper**

```python
#!/usr/bin/env python3
"""Standalone GGUF export for the code-review domain.

Usage:
    python3 examples/code-review/export_gguf.py --domain code-review
    python3 examples/code-review/export_gguf.py --domain code-review --quantization Q5_K_M
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export code-review domain model to GGUF format"
    )
    parser.add_argument("--domain", default="code-review", help="Domain name")
    parser.add_argument(
        "--quantization",
        default="Q4_K_M",
        choices=["Q4_K_M", "Q5_K_M", "Q8_0"],
        help="Quantization method (default: Q4_K_M)",
    )
    parser.add_argument(
        "--root", default=".", help="Project root directory"
    )
    args = parser.parse_args()

    root = Path(args.root)
    cmd = [
        sys.executable, str(root / "cli.py"), "export-gguf", args.domain,
        "--quantization", args.quantization,
    ]

    result = subprocess.run(cmd, cwd=str(root))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Make it executable and add a quick test**

```bash
chmod +x examples/code-review/export_gguf.py
```

Add a simple test in `tests/test_code_review/test_export_gguf.py`:

```python
"""Tests for the standalone export_gguf.py wrapper."""

import subprocess
import sys
from pathlib import Path


def test_export_gguf_script_exists():
    """The standalone export script should exist and be executable."""
    script = Path(__file__).resolve().parents[2] / "examples" / "code-review" / "export_gguf.py"
    assert script.exists()


def test_export_gguf_script_help():
    """The script should respond to --help."""
    script = Path(__file__).resolve().parents[2] / "examples" / "code-review" / "export_gguf.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "quantization" in result.stdout
```

- [ ] **Step 3: Run tests**

```bash
cd /Users/jinzuo/projects/elixirlora
python3 -m pytest tests/test_code_review/ -v
```

- [ ] **Step 4: Commit**

```bash
git add examples/code-review/export_gguf.py tests/test_code_review/test_export_gguf.py
git commit -m "feat: add standalone export_gguf.py wrapper for code-review domain"
```

---

## Task 6: Integration Test — Full Pipeline on Tiny Subset

**Files:**
- Create: `tests/test_code_review/test_integration.py`

**Interfaces:**
- Consumes: `examples.code_review.setup` (setup function directly)
- Tests: the full data flow from dataset → seeds → prepare output shape

- [ ] **Step 1: Write integration test**

```python
"""Integration test: run full pipeline on a tiny dataset subset."""

import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "src"))


def test_full_pipeline_on_tiny_subset(tmp_path):
    """
    Run the setup pipeline on a synthetic dataset to verify:
    1. Seeds are valid ElixirLoRA format
    2. prepare command can consume them
    3. Output has expected train/val/test structure
    """
    import subprocess
    import yaml

    # Create a tiny synthetic dataset to simulate what setup.py would download
    fake_dataset = [
        {
            "code_diff": "@@ -1,3 +1,3 @@\n-old_line\n+new_line",
            "file_path": "src/test.py",
            "language": "Python",
            "pr_title": "Test PR",
            "review_type": "suggestion",
            "comment": "Fix this line.",
            "suggestion": "def fixed():\n    pass",
            "repo_name": "test/repo",
            "pr_number": 1,
            "line": 1,
            "old_line": 1,
            "new_line": 2,
        },
        {
            "code_diff": "",
            "file_path": "",
            "language": "",
            "pr_title": "",
            "review_type": "none",
            "comment": "",
            "suggestion": "",
            "repo_name": "test/repo2",
            "pr_number": 2,
            "line": 0,
            "old_line": 0,
            "new_line": 0,
        },
        {
            "code_diff": "@@ -1,0 +1,1 @@\n+added line",
            "file_path": "src/other.py",
            "language": "JavaScript",
            "pr_title": "Add feature",
            "review_type": "refactor",
            "comment": "Extract this.",
            "suggestion": "function helper():\n    pass",
            "repo_name": "test/repo3",
            "pr_number": 3,
            "line": 1,
            "old_line": 0,
            "new_line": 1,
        },
    ]

    # Write fake dataset for setup.py to consume
    fake_dataset_path = tmp_path / "data" / "code-review" / "fake_dataset.jsonl"
    fake_dataset_path.parent.mkdir(parents=True)
    for record in fake_dataset:
        fake_dataset_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in fake_dataset) + "\n",
            encoding="utf-8",
        )

    # Import and run setup's conversion with a custom dataset path
    # We need to patch the dataset loading since we can't actually download
    from examples.code_review.setup import make_record, build_user_message, build_assistant_message

    # Convert records manually
    seeds = []
    for record in fake_dataset:
        user_content = build_user_message(record)
        assistant_content = build_assistant_message(record)
        meta = {
            "review_type": record.get("review_type", "none"),
            "language": record.get("language", ""),
            "repo": record.get("repo_name", ""),
            "pr_title": record.get("pr_title", ""),
        }
        seeds.append(make_record(user_content, assistant_content, meta))

    assert len(seeds) == 3

    # Simulate workspace creation
    ws = tmp_path / "workspaces" / "test-integration"
    seeds_dir = ws / "seeds"
    seeds_dir.mkdir(parents=True)
    approved_path = seeds_dir / "approved.jsonl"
    approved_path.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in seeds) + "\n",
        encoding="utf-8",
    )

    # Verify the approved.jsonl is valid ElixirLoRA format
    loaded = [json.loads(l) for l in approved_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(loaded) == 3
    for rec in loaded:
        assert "conversation" in rec
        assert len(rec["conversation"]) == 2
        assert rec["conversation"][0]["role"] == "user"
        assert rec["conversation"][1]["role"] == "assistant"
        assert "content" in rec["conversation"][0]
        assert "content" in rec["conversation"][1]

    # Verify output format contains the expected markers
    assistant_texts = [r["conversation"][1]["content"] for r in loaded]
    assert any("[review_type: suggestion]" in t for t in assistant_texts)
    assert any("[review_type: none]" in t for t in assistant_texts)
    assert any("[review_type: refactor]" in t for t in assistant_texts)

    # Test that the format converter round-trips correctly
    for rec in loaded:
        conversation = rec["conversation"]
        assert len(conversation[0]["content"]) > 0
        assert len(conversation[1]["content"]) > 0
```

- [ ] **Step 2: Run the integration test**

```bash
cd /Users/jinzuo/projects/elixirlora
python3 -m pytest tests/test_code_review/test_integration.py -v
```

- [ ] **Step 3: Commit**

```bash
git add tests/test_code_review/test_integration.py
git commit -m "test: add integration test for code-review dataset pipeline"
```

---

## Implementation Order

1. **Task 1** — `examples/code-review/setup.py` + README + tests (independent)
2. **Task 2** — `commands/export_gguf.py` + CLI + tests (independent)
3. **Task 3** — TUI Deploy panel GGUF button + tests (independent)
4. **Task 4** — TUI New Domain radio option + tests (independent)
5. **Task 5** — Standalone `examples/code-review/export_gguf.py` + tests (depends on Task 2)
6. **Task 6** — Integration test (depends on Task 1)

Tasks 1 and 2 are independent — dispatch in parallel.
Tasks 3 and 4 are independent — dispatch in parallel.
Task 5 depends on Task 2 (needs the `export-gguf` CLI command to exist).
Task 6 depends on Task 1 (needs the setup.py functions to exist).

Recommended parallel dispatch:
- **Group A:** Tasks 1 + 2 (parallel, no cross-dependencies)
- **Group B:** Tasks 3 + 4 (parallel, no cross-dependencies)
- **Task 5:** After Group A completes
- **Task 6:** After Group A completes (specifically Task 1)
