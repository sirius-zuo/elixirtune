import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
import yaml
from data.synthetic.io import read_jsonl
from data.preprocessor import DataPreprocessor

app = typer.Typer(context_settings={"allow_interspersed_args": True})

from commands import _ws

_DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant."


def _resolve_system_prompt(domain: str) -> str:
    """Use the domain's configured prepare.system_prompt, else a generic default."""
    cfg_path = _ws(domain) / "config.yaml"
    if cfg_path.exists():
        data = yaml.safe_load(cfg_path.read_text()) or {}
        sp = (data.get("prepare") or {}).get("system_prompt")
        if sp:
            return sp
    return _DEFAULT_SYSTEM_PROMPT


@app.callback(invoke_without_command=True)
def prepare(
    ctx: typer.Context,
    domain: str = typer.Argument(...),
    system_prompt: str = typer.Option(None, help="System prompt (default: config prepare.system_prompt or generic)"),
    out_dir: str = typer.Option(None, help="Output dir (default: workspaces/<domain>/processed)"),
    test_split: float = typer.Option(0.1, help="Fraction held out for test"),
    val_split: float = typer.Option(0.1, help="Fraction held out for validation"),
):
    """Convert filtered JSONL (or seeds) into train/val/test splits."""
    if ctx.invoked_subcommand is not None:
        return
    if system_prompt is None:
        system_prompt = _resolve_system_prompt(domain)
    filtered = _ws(domain) / "generated" / "filtered.jsonl"
    seeds = _ws(domain) / "seeds" / "approved.jsonl"
    # Curated seeds are always included; generated data is added when present.
    seed_recs = read_jsonl(seeds) if seeds.exists() else []
    gen_recs = read_jsonl(filtered) if filtered.exists() else []

    # Remove exact-duplicate conversations (e.g. a generated copy of a seed).
    records, seen = [], set()
    for rec in seed_recs + gen_recs:
        key = json.dumps(rec["conversation"], sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        records.append(rec)

    if not records:
        typer.echo(
            f"No data found. Expected seeds at {seeds} or generated at {filtered}. "
            f"Run: curate {domain} (and optionally generate {domain})",
            err=True,
        )
        raise typer.Exit(1)
    typer.echo(
        f"Using {len(seed_recs)} seed + {len(gen_recs)} generated "
        f"= {len(records)} unique records."
    )

    preprocessor = DataPreprocessor(
        system_prompt=system_prompt,
        test_split_ratio=test_split,
        val_split_ratio=val_split,
    )

    samples = []
    for rec in records:
        conversation = rec["conversation"]
        for i in range(0, len(conversation) - 1, 2):
            q = conversation[i]["content"]
            a = conversation[i + 1]["content"]
            samples.append(preprocessor.format_conversation_sample(q, a))

    random.shuffle(samples)
    train, val, test = preprocessor.create_train_val_test_split(samples)

    out = Path(out_dir) if out_dir else _ws(domain) / "processed"
    out.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        (out / f"{name}.json").write_text(json.dumps(split, indent=2))
    stats = {"train_size": len(train), "val_size": len(val), "test_size": len(test),
             "total_size": len(samples)}
    (out / "data_stats.json").write_text(json.dumps(stats, indent=2))
    typer.echo(
        f"Prepared {len(samples)} samples → {out} "
        f"(train={len(train)}, val={len(val)}, test={len(test)})"
    )
