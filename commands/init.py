import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
from data.synthetic.io import read_jsonl, write_jsonl

app = typer.Typer()

def _ws(domain: str) -> Path:
    return Path("workspaces") / domain

@app.command()
def init(domain: str, desc: str = typer.Option(None), seeds: str = typer.Option(None)):
    """Initialise a new domain workspace."""
    cand = _ws(domain) / "seeds" / "candidates.jsonl"
    cand.parent.mkdir(parents=True, exist_ok=True)
    if seeds:
        recs = read_jsonl(seeds)
        write_jsonl(cand, recs)
        typer.echo(f"Imported {len(recs)} seeds to {cand}")
    else:
        cand.touch()
        typer.echo(f"Created empty seed file at {cand}")
    if desc:
        (_ws(domain) / "description.txt").write_text(desc)
