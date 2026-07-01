import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer

app = typer.Typer(context_settings={"allow_interspersed_args": True})

from commands import _ws


@app.callback(invoke_without_command=True)
def evaluate(
    ctx: typer.Context,
    domain: str = typer.Argument(...),
    method: str = typer.Option("lm", help="Evaluation method: lm | embedding | cross-encoder"),
    eval_config: Path = typer.Option(None, help="[lm] Path to runtime eval config YAML"),
    adapters_path: Path = typer.Option(None, help="Path to adapters dir"),
    test_data: Path = typer.Option(None, help="Path to test data file"),
    fused_path: Path = typer.Option(None, help="[lm] Path to fused model dir"),
    model_config: Path = typer.Option(None, help="Path to model config YAML"),
    max_samples: int = typer.Option(100, help="[lm] Max test samples (default 100)"),
    beir_dataset: str = typer.Option(None, help="[embedding] BEIR dataset name (e.g. scifact)"),
    val_data: Path = typer.Option(None, help="[embedding] Path to embedding_val.json"),
) -> None:
    """Evaluate a trained model."""
    if ctx.invoked_subcommand is not None:
        return

    ws = _ws(domain)

    if method == "lm":
        import yaml
        from evaluation.evaluator import ModelEvaluator

        if test_data is None:
            test_data = ws / "processed" / "test.json"
        cfg_path = Path(model_config) if model_config else ws / "runtime_model_config.yaml"
        if not cfg_path.exists():
            typer.echo(
                f"Model config not found at {cfg_path}. "
                "Pass --model-config <path> or run the TUI first.",
                err=True,
            )
            raise typer.Exit(1)
        model_cfg = yaml.safe_load(cfg_path.read_text())
        base_model = model_cfg["base_model"]["path"]
        evaluator = ModelEvaluator(str(eval_config))
        if adapters_path and Path(adapters_path).exists():
            evaluator.comprehensive_model_comparison(
                base_model_path=base_model,
                adapter_path=str(adapters_path),
                fused_model_path=str(fused_path) if fused_path else None,
                test_data_path=str(test_data),
                max_samples=max_samples,
            )
        else:
            evaluator.evaluate_model_from_path(base_model, "base_model", str(test_data), max_samples=max_samples)

    elif method in ("embedding", "cross-encoder"):
        import yaml
        from evaluation.embedding_evaluator import recall_at_k, run_beir

        cfg_path = Path(model_config) if model_config else ws / "runtime_model_config.yaml"
        model_cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
        adapters_dir = Path(adapters_path) if adapters_path else ws / "adapters"

        if method == "embedding":
            from mlx_tune.embeddings import FastEmbeddingModel
            em_cfg = model_cfg.get("embedding", {})
            loaded_model, tokenizer = FastEmbeddingModel.from_pretrained(
                em_cfg.get("base_model", "mlx-community/all-MiniLM-L6-v2"),
                max_seq_length=em_cfg.get("max_seq_length", 512),
            )
        else:
            from sentence_transformers import CrossEncoder
            ce_cfg = model_cfg.get("cross_encoder", {})
            ce_path = str(ws / "ce_adapters") if (ws / "ce_adapters").exists() else ce_cfg.get("base_model", "")
            loaded_model, tokenizer = CrossEncoder(ce_path), None

        val_path = val_data or ws / "processed" / "embedding_val.json"
        if val_path.exists() and method == "embedding":
            metrics = recall_at_k(val_path, loaded_model, tokenizer)
            for key, val in metrics.items():
                typer.echo(f"{key}: {val:.4f}")

        if beir_dataset and method == "embedding":
            beir_result = run_beir(beir_dataset, loaded_model, tokenizer)
            for key, val in beir_result.items():
                typer.echo(f"BEIR {beir_dataset} {key}: {val:.4f}")
    else:
        typer.echo(f"Unknown method '{method}'. Choose: lm, embedding, cross-encoder", err=True)
        raise typer.Exit(1)
