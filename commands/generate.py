import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
from data.synthetic.config import load_config
from data.synthetic.teacher import from_config
from data.synthetic.embedder import SentenceTransformerEmbedder
from data.synthetic.pipeline import run_generate
from data.synthetic.io import read_jsonl

app = typer.Typer()

def _ws(domain: str) -> Path:
    return Path("workspaces") / domain

@app.command()
def generate(domain: str):
    """Generate and filter synthetic training data."""
    cfg = load_config(_ws(domain) / "config.yaml")
    teacher = from_config(cfg)
    embedder = SentenceTransformerEmbedder()
    seeds = read_jsonl(_ws(domain) / "seeds" / "approved.jsonl")
    run_generate(domain, cfg, teacher, embedder, seeds)
