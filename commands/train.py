import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer

app = typer.Typer(context_settings={"allow_interspersed_args": True})


@app.callback(invoke_without_command=True)
def train(
    ctx: typer.Context,
    domain: str = typer.Argument(...),
    method: str = typer.Option("sft", help="Training method: sft | dpo | grpo"),
    model_config: Path = typer.Option(..., help="Path to runtime model config YAML"),
    training_config: Path = typer.Option(..., help="Path to runtime training config YAML"),
    train_data: Path = typer.Option(..., help="Path to train.json"),
    val_data: Path = typer.Option(None, help="Path to val.json (optional)"),
) -> None:
    """Fine-tune a model using the specified training method."""
    if ctx.invoked_subcommand is not None:
        return
    if method == "sft":
        from src.training.sft import run
    elif method == "dpo":
        from src.training.dpo import run
    elif method == "grpo":
        from src.training.grpo import run
    else:
        raise typer.BadParameter(f"Unknown method '{method}'. Choose: sft, dpo, grpo")
    run(domain, model_config, training_config, train_data, val_data)
