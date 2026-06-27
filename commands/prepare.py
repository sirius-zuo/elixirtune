import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
from data.synthetic.io import read_jsonl
from data.preprocessor import DataPreprocessor

app = typer.Typer(context_settings={"allow_interspersed_args": True})

from commands import _ws

@app.callback(invoke_without_command=True)
def prepare(
    ctx: typer.Context,
    domain: str = typer.Argument(...),
    system_prompt: str = typer.Option(..., help="System prompt for the fine-tuned model"),
    out_dir: str = typer.Option(None, help="Output dir (default: workspaces/<domain>/processed)"),
    test_split: float = typer.Option(0.1, help="Fraction held out for test"),
    val_split: float = typer.Option(0.1, help="Fraction held out for validation"),
):
    """Convert filtered JSONL from generate into train/val/test splits."""
    if ctx.invoked_subcommand is not None:
        return
    filtered = _ws(domain) / "generated" / "filtered.jsonl"
    if not filtered.exists():
        typer.echo(f"No filtered data at {filtered}. Run: generate {domain}", err=True)
        raise typer.Exit(1)

    preprocessor = DataPreprocessor(
        system_prompt=system_prompt,
        test_split_ratio=test_split,
        val_split_ratio=val_split,
    )

    samples = []
    for rec in read_jsonl(filtered):
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
