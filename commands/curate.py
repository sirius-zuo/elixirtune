import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
from data.synthetic.io import read_jsonl

app = typer.Typer(context_settings={"allow_interspersed_args": True})

from commands import _ws

@app.callback(invoke_without_command=True)
def curate(ctx: typer.Context, domain: str = typer.Argument(...)):
    """Curate seed examples for a domain."""
    if ctx.invoked_subcommand is not None:
        return
    cand = _ws(domain) / "seeds" / "candidates.jsonl"
    approved = _ws(domain) / "seeds" / "approved.jsonl"
    seeds = read_jsonl(cand) if cand.exists() else []
    if not seeds:
        typer.echo(
            f"candidates.jsonl is empty — add seed examples to {cand} first.",
            err=True,
        )
        raise typer.Exit(1)
    shutil.copyfile(cand, approved)
    typer.echo(f"Approved {len(seeds)} seeds → {approved}")
