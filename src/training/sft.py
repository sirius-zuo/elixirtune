import json
import yaml
from pathlib import Path
from datasets import Dataset

from src.training.metrics_writer import MetricsWriterCallback


def _load_configs(model_config_path: Path, training_config_path: Path) -> dict:
    m = yaml.safe_load(model_config_path.read_text())
    t = yaml.safe_load(training_config_path.read_text())
    return {"model": m, "training": t}


def run(
    domain: str,
    model_config_path: Path,
    training_config_path: Path,
    train_data_path: Path,
    val_data_path: Path | None,
) -> None:
    from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig

    cfg = _load_configs(Path(model_config_path), Path(training_config_path))
    m_cfg = cfg["model"]
    t_cfg = cfg["training"]

    metrics_path = (
        Path("workspaces") / domain / "logs" / "training" / "training_metrics.json"
    )
    output_dir = str(Path("workspaces") / domain / "adapters")

    model, tokenizer = FastLanguageModel.from_pretrained(
        m_cfg["base_model"]["path"],
        max_seq_length=m_cfg.get("max_seq_length", 2048),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=m_cfg["lora"]["rank"],
        target_modules=m_cfg["lora"].get("keys", ["q_proj", "v_proj"]),
        lora_alpha=m_cfg["lora"]["scale"],
        lora_dropout=m_cfg["lora"]["dropout"],
    )

    train_ds = Dataset.from_list(json.loads(Path(train_data_path).read_text()))
    eval_ds = (
        Dataset.from_list(json.loads(Path(val_data_path).read_text()))
        if val_data_path
        else None
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(
            output_dir=output_dir,
            per_device_train_batch_size=t_cfg["training"]["batch_size"],
            learning_rate=t_cfg["training"]["learning_rate"],
            max_steps=t_cfg["training"]["iters"],
            eval_steps=t_cfg["training"]["steps_per_eval"],
        ),
        callbacks=[MetricsWriterCallback(metrics_path)],
    )
    trainer.train()
