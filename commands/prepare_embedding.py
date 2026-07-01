import json
import random
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
from commands import _ws

app = typer.Typer(context_settings={"allow_interspersed_args": True})

_DEFAULT_ANCHOR_COL = "anchor"
_DEFAULT_POSITIVE_COL = "positive"


def _read_data_file(path: Path) -> list[dict]:
    text = path.read_text()
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _validate_columns(records: list[dict], anchor_col: str, positive_col: str) -> None:
    missing = [c for c in (anchor_col, positive_col) if c not in records[0]]
    if missing:
        typer.echo(
            f"Data file missing required columns: {missing}. "
            f"Expected at least '{anchor_col}' and '{positive_col}'.",
            err=True,
        )
        raise typer.Exit(1)


def _split_and_write(records: list[dict], ws: Path, val_split: float) -> None:
    random.shuffle(records)
    val_n = max(1, int(len(records) * val_split))
    val, train = records[:val_n], records[val_n:]
    out = ws / "processed"
    out.mkdir(parents=True, exist_ok=True)
    (out / "embedding_train.json").write_text(json.dumps(train, indent=2))
    (out / "embedding_val.json").write_text(json.dumps(val, indent=2))
    typer.echo(
        f"Prepared {len(records)} pairs → {out} "
        f"(train={len(train)}, val={len(val)})"
    )


@app.callback(invoke_without_command=True)
def prepare_embedding(
    ctx: typer.Context,
    domain: str = typer.Argument(...),
    mode: str = typer.Option("import", help="Preparation mode: import | convert"),
    data_file: Path = typer.Option(None, help="[import] Path to JSON/JSONL with anchor/positive pairs"),
    val_split: float = typer.Option(0.1, help="Fraction held out for validation"),
    anchor_column: str = typer.Option(_DEFAULT_ANCHOR_COL, help="Column name for anchor texts"),
    positive_column: str = typer.Option(_DEFAULT_POSITIVE_COL, help="Column name for positive texts"),
) -> None:
    """Prepare anchor/positive(/negative) pair data for embedding fine-tuning."""
    if ctx.invoked_subcommand is not None:
        return

    ws = _ws(domain)

    if mode == "import":
        if not data_file:
            typer.echo("--data-file is required for import mode.", err=True)
            raise typer.Exit(1)
        records = _read_data_file(data_file)
        if not records:
            typer.echo("Data file is empty.", err=True)
            raise typer.Exit(1)
        _validate_columns(records, anchor_column, positive_column)
        # Copy raw file for provenance
        raw_dir = ws / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / data_file.name).write_bytes(data_file.read_bytes())
        _split_and_write(records, ws, val_split)

    elif mode == "convert":
        from data.synthetic.io import read_jsonl
        seeds_path = ws / "seeds" / "approved.jsonl"
        generated_path = ws / "generated" / "filtered.jsonl"
        seed_recs = read_jsonl(str(seeds_path)) if seeds_path.exists() else []
        gen_recs = read_jsonl(str(generated_path)) if generated_path.exists() else []
        all_recs = seed_recs + gen_recs
        if not all_recs:
            typer.echo(
                f"No source data found. Add seeds to {seeds_path} first.",
                err=True,
            )
            raise typer.Exit(1)
        pairs = []
        for rec in all_recs:
            conv = rec.get("conversation", [])
            for i in range(0, len(conv) - 1, 2):
                if conv[i].get("role") == "user" and conv[i + 1].get("role") == "assistant":
                    pairs.append({
                        anchor_column: conv[i]["content"],
                        positive_column: conv[i + 1]["content"],
                    })
        if not pairs:
            typer.echo("No Q&A pairs extracted from source data.", err=True)
            raise typer.Exit(1)
        typer.echo(f"Extracted {len(pairs)} pairs from {len(all_recs)} records.")
        _split_and_write(pairs, ws, val_split)

    else:
        typer.echo(f"Unknown mode '{mode}'. Choose: import, convert", err=True)
        raise typer.Exit(1)
