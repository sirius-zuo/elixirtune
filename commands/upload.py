import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer

app = typer.Typer(context_settings={"allow_interspersed_args": True})

def _ws(domain: str) -> Path:
    return Path("workspaces") / domain

@app.callback(invoke_without_command=True)
def upload(
    ctx: typer.Context,
    domain: str = typer.Argument(...),
    repo_name: str = typer.Option(..., help="HuggingFace repo (username/repo-name)"),
    private: bool = typer.Option(False, help="Make repository private"),
    token: str = typer.Option(None, envvar="HF_TOKEN", help="HuggingFace token"),
):
    """Upload fused model to HuggingFace Hub."""
    if ctx.invoked_subcommand is not None:
        return
    import huggingface_hub

    fused = _ws(domain) / "fused"
    if not fused.exists() or not any(fused.iterdir()):
        typer.echo(f"No fused model at {fused}. Run: fuse {domain} first.", err=True)
        raise typer.Exit(1)

    if not token:
        typer.echo("HuggingFace token required. Set HF_TOKEN or use --token.", err=True)
        raise typer.Exit(1)

    typer.echo(f"Creating repository {repo_name}...")
    huggingface_hub.create_repo(
        repo_id=repo_name, private=private, token=token, exist_ok=True, repo_type="model"
    )
    typer.echo(f"Uploading {fused}...")
    huggingface_hub.upload_folder(
        folder_path=str(fused),
        repo_id=repo_name,
        token=token,
        commit_message=f"Upload fused model for domain: {domain}",
    )
    typer.echo(f"Done. https://huggingface.co/{repo_name}")
