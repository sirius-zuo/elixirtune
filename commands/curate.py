import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
from data.synthetic.io import read_jsonl

app = typer.Typer()

def _ws(domain: str) -> Path:
    return Path("workspaces") / domain

@app.command()
def curate(domain: str):
    """Curate seed examples for a domain."""
    cand = _ws(domain) / "seeds" / "candidates.jsonl"
    approved = _ws(domain) / "seeds" / "approved.jsonl"
    shutil.copyfile(cand, approved)
    typer.echo(f"Approved {len(read_jsonl(approved))} seeds → {approved}")
