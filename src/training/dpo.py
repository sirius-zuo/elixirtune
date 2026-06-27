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
    if not data or not isinstance(data[0], dict) or "chosen" not in data[0]:
        raise ValueError(
            "DPO requires training data with fields {prompt, chosen, rejected}. "
            "Run the DPO data preparation pipeline first."
        )
    from mlx_tune import DPOTrainer, DPOConfig
    raise NotImplementedError("DPO training pipeline not yet configured.")
