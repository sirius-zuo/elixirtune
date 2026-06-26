import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import typer
from data.synthetic.config import load_config
from data.synthetic.bootstrap import bootstrap_seeds
from data.synthetic.teacher import from_config
from data.synthetic.embedder import SentenceTransformerEmbedder
from data.synthetic.pipeline import run_generate
from data.synthetic.io import read_jsonl, write_jsonl

app = typer.Typer()


def _ws(domain: str) -> Path:
    return Path("workspaces") / domain


@app.command()
def init(domain: str, desc: str = typer.Option(None), seeds: str = typer.Option(None)):
    cand = _ws(domain) / "seeds" / "candidates.jsonl"
    cand.parent.mkdir(parents=True, exist_ok=True)
    if seeds:
        write_jsonl(cand, read_jsonl(seeds))
        typer.echo(f"Imported {len(read_jsonl(cand))} seeds to {cand}")
    else:
        if not desc:
            raise typer.BadParameter("provide --seeds PATH or --desc TEXT")
        cfg = load_config(domain)
        teacher = from_config(cfg)
        recs = bootstrap_seeds(desc, teacher, cfg["bootstrap"]["starter_count"])
        write_jsonl(cand, recs)
        typer.echo(f"Bootstrapped {len(recs)} candidate seeds to {cand}. Edit, then run: curate {domain}")


@app.command()
def curate(domain: str):
    cand = _ws(domain) / "seeds" / "candidates.jsonl"
    approved = _ws(domain) / "seeds" / "approved.jsonl"
    shutil.copyfile(cand, approved)
    typer.echo(f"Approved {len(read_jsonl(approved))} seeds → {approved}")


@app.command()
def generate(domain: str):
    cfg = load_config(domain)
    teacher = from_config(cfg)
    embedder = SentenceTransformerEmbedder(cfg["filter"]["dedup"]["embedding_model"])
    run_dir = run_generate(domain, cfg, teacher, embedder)
    typer.echo(f"Done. Run artifacts in {run_dir}")


if __name__ == "__main__":
    app()
