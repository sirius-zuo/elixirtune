import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
from data.synthetic.io import read_jsonl, write_jsonl

app = typer.Typer(context_settings={"allow_interspersed_args": True})

def _ws(domain: str) -> Path:
    return Path("workspaces") / domain

@app.callback(invoke_without_command=True)
def init(ctx: typer.Context, domain: str = typer.Argument(...), desc: str = typer.Option(None), seeds: str = typer.Option(None)):
    """Initialise a new domain workspace."""
    if ctx.invoked_subcommand is not None:
        return
    cand = _ws(domain) / "seeds" / "candidates.jsonl"
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
        (_ws(domain) / "description.txt").write_text(desc)
