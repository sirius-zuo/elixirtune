import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
from data.synthetic.config import load_config
from data.synthetic.teacher import from_config
from data.synthetic.embedder import SentenceTransformerEmbedder
from data.synthetic.pipeline import run_generate, GenerationEmptyError

app = typer.Typer(context_settings={"allow_interspersed_args": True})

from commands import _ws

@app.callback(invoke_without_command=True)
def generate(
    ctx: typer.Context,
    domain: str = typer.Argument(...),
    verbose: bool = typer.Option(False, "--verbose", help="Log each request/response and per-item judge score"),
):
    """Generate and filter synthetic training data."""
    if ctx.invoked_subcommand is not None:
        return
    cfg = load_config(domain)
    teacher = from_config(cfg)
    embedder = SentenceTransformerEmbedder(cfg["filter"]["dedup"]["embedding_model"])
    try:
        run_generate(domain, cfg, teacher, embedder, verbose=verbose)
    except GenerationEmptyError:
        raise typer.Exit(1)
