import json
import yaml
from pathlib import Path

import src._compat  # noqa: F401 — apply Python 3.14 / datasets compat patches
from datasets import Dataset

from .metrics_writer import MetricsWriterCallback


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
        r=int(m_cfg["lora"]["rank"]),
        target_modules=m_cfg["lora"].get("keys", ["q_proj", "v_proj"]),
        lora_alpha=float(m_cfg["lora"]["scale"]),
        lora_dropout=float(m_cfg["lora"]["dropout"]),
    )

    def _load(path: Path) -> Dataset:
        raw = json.loads(path.read_text())
        records = raw if isinstance(raw[0], dict) else [{"text": s} for s in raw]
        return Dataset.from_list(records)

    train_ds = _load(Path(train_data_path))
    eval_ds = _load(Path(val_data_path)) if val_data_path else None

    sft_args = dict(
        output_dir=output_dir,
        per_device_train_batch_size=int(t_cfg["training"]["batch_size"]),
        learning_rate=float(t_cfg["training"]["learning_rate"]),
        max_steps=int(t_cfg["training"]["iters"]),
        dataset_text_field="text",
    )
    if eval_ds is not None:
        sft_args["eval_steps"] = int(t_cfg["training"]["steps_per_eval"])

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=SFTConfig(**sft_args),
        callbacks=[MetricsWriterCallback(metrics_path)],
    )
    trainer.train()
