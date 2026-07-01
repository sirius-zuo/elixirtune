import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
import yaml
from data.synthetic.io import read_jsonl, write_jsonl
from commands import _ws

app = typer.Typer(context_settings={"allow_interspersed_args": True})

@app.callback(invoke_without_command=True)
def init(
    ctx: typer.Context,
    domain: str = typer.Argument(...),
    desc: str = typer.Option(None),
    seeds: str = typer.Option(None),
    type: str = typer.Option("lm", help="Domain type: lm | embedding"),
):
    """Initialise a new domain workspace."""
    if ctx.invoked_subcommand is not None:
        return
    if type not in ("lm", "embedding"):
        typer.echo(f"Invalid type '{type}'. Choose: lm, embedding", err=True)
        raise typer.Exit(1)
    ws = _ws(domain)
    ws.mkdir(parents=True, exist_ok=True)
    cand = ws / "seeds" / "candidates.jsonl"
    cand.parent.mkdir(parents=True, exist_ok=True)
    if seeds:
        recs = read_jsonl(seeds)
        write_jsonl(cand, recs)
        typer.echo(f"Imported {len(recs)} seeds to {cand}")
    else:
        cand.touch()
        typer.echo(f"Created empty seed file at {cand}")
        typer.echo("Add seeds to the file or re-run with --seeds <path>", err=True)
    if desc:
        (ws / "description.txt").write_text(desc)
    cfg_path = ws / "config.yaml"
    existing = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    existing["type"] = type
    cfg_path.write_text(yaml.safe_dump(existing))
