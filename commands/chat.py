import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
import yaml

app = typer.Typer(context_settings={"allow_interspersed_args": True})


def _ws(domain: str) -> Path:
    return Path("workspaces") / domain


def _system_prompt(domain: str) -> str:
    cfg_path = _ws(domain) / "config.yaml"
    if cfg_path.exists():
        try:
            data = yaml.safe_load(cfg_path.read_text()) or {}
            sp = data.get("chat", {}).get("system_prompt") if isinstance(data, dict) else None
            if sp:
                return sp
        except Exception:
            pass
    return "You are a helpful assistant."


@app.callback(invoke_without_command=True)
def chat(
    ctx: typer.Context,
    domain: str = typer.Argument(...),
    fused: bool = typer.Option(True, help="Use fused model (default) or runtime adapters"),
    max_tokens: int = typer.Option(200),
    temperature: float = typer.Option(0.7),
) -> None:
    """Start an interactive chat session with the domain's fine-tuned model."""
    if ctx.invoked_subcommand is not None:
        return
    from inference.chat_interface import ChatInterface

    ws = _ws(domain)
    model_path = str(ws / "fused") if fused else str(ws / "adapters")
    if not Path(model_path).exists():
        typer.echo(f"Model not found at {model_path}. Run fuse first.", err=True)
        raise typer.Exit(1)

    system_prompt = _system_prompt(domain)
    interface = ChatInterface(model_path, system_prompt)
    interface.start_chat(max_tokens=max_tokens, temperature=temperature)
