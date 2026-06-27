import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer
import yaml

app = typer.Typer(context_settings={"allow_interspersed_args": True})

from commands import _ws


@app.callback(invoke_without_command=True)
def fuse(
    ctx: typer.Context,
    domain: str = typer.Argument(...),
    model_config: Path = typer.Option(..., help="Path to runtime model config YAML"),
    output_path: Path = typer.Option(None, help="Fused model output path (default: workspaces/<domain>/fused)"),
    eval_config: Path = typer.Option(None, help="Eval config YAML (omit to skip post-fuse eval)"),
    test_data: Path = typer.Option(None, help="Test data for post-fuse eval"),
    adapters_path: Path = typer.Option(None, help="Adapters dir (default: workspaces/<domain>/adapters)"),
) -> None:
    """Fuse LoRA adapters into the base model and optionally evaluate the result."""
    if ctx.invoked_subcommand is not None:
        return
    from utils.fusion import AdapterFusion

    ws = _ws(domain)
    m_cfg = yaml.safe_load(Path(model_config).read_text())
    base_model = m_cfg["base_model"]["path"]

    adapters = Path(adapters_path) if adapters_path else ws / "adapters"
    out = Path(output_path) if output_path else ws / "fused"

    fusion = AdapterFusion()
    if not fusion.validate_fusion_inputs(base_model, str(adapters)):
        raise typer.Exit(1)

    fusion.fuse_adapters(base_model, str(adapters), str(out))
    typer.echo(f"Fused model saved to: {out}")

    if eval_config and Path(eval_config).exists():
        from evaluation.evaluator import ModelEvaluator
        test = Path(test_data) if test_data else ws / "processed" / "test.json"
        evaluator = ModelEvaluator(str(eval_config))
        evaluator.evaluate_model_from_path(str(out), "lora_fused", str(test))
