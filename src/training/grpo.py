from pathlib import Path


def run(
    domain: str,
    model_config_path: Path,
    training_config_path: Path,
    train_data_path: Path,
    val_data_path: Path | None,
) -> None:
    import json
    data = json.loads(Path(train_data_path).read_text()) if Path(train_data_path).exists() else []
    if not data or not isinstance(data[0], dict) or "prompt" not in data[0]:
        raise ValueError(
            "GRPO requires training data with a {prompt} field and a configured reward function. "
            "Run the GRPO data preparation pipeline first."
        )
    from mlx_tune import GRPOTrainer, GRPOConfig
    raise NotImplementedError("GRPO training pipeline not yet configured.")
