import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer

app = typer.Typer(context_settings={"allow_interspersed_args": True})

def _ws(domain: str) -> Path:
    return Path("workspaces") / domain


@app.callback(invoke_without_command=True)
def evaluate(
    ctx: typer.Context,
    domain: str = typer.Argument(...),
    eval_config: Path = typer.Option(..., help="Path to runtime eval config YAML"),
    adapters_path: Path = typer.Option(None, help="Path to adapters dir (omit to eval base only)"),
    test_data: Path = typer.Option(None, help="Path to test.json (default: workspaces/<domain>/processed/test.json)"),
    fused_path: Path = typer.Option(None, help="Path to fused model dir (optional)"),
) -> None:
    """Evaluate base model and/or fine-tuned adapters for a domain."""
    if ctx.invoked_subcommand is not None:
        return
    from evaluation.evaluator import ModelEvaluator
    import yaml

    ws = _ws(domain)
    if test_data is None:
        test_data = ws / "processed" / "test.json"

    model_cfg = yaml.safe_load((ws / "runtime_model_config.yaml").read_text())
    base_model = model_cfg["base_model"]["path"]

    evaluator = ModelEvaluator(str(eval_config))

    if adapters_path and Path(adapters_path).exists():
        evaluator.comprehensive_model_comparison(
            base_model_path=base_model,
            adapter_path=str(adapters_path),
            fused_model_path=str(fused_path) if fused_path else None,
            test_data_path=str(test_data),
        )
    else:
        evaluator.evaluate_model_from_path(base_model, "base_model", str(test_data))
