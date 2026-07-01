import json
import yaml
from pathlib import Path

import src._compat  # noqa: F401 — apply Python 3.14 / datasets compat patches


def _load_configs(model_config_path: Path, training_config_path: Path) -> dict:
    m = yaml.safe_load(Path(model_config_path).read_text())
    t = yaml.safe_load(Path(training_config_path).read_text())
    return {"model": m, "training": t}


def run(
    domain: str,
    model_config_path: Path,
    training_config_path: Path,
    train_data_path: Path,
    val_data_path: Path | None,
) -> None:
    from mlx_tune.embeddings import FastEmbeddingModel, EmbeddingSFTTrainer, EmbeddingSFTConfig
    from datasets import Dataset

    cfg = _load_configs(Path(model_config_path), Path(training_config_path))
    m_cfg = cfg["model"]["embedding"]
    t_cfg = cfg["training"]["embedding"]

    output_dir = str(Path("workspaces") / domain / "adapters")

    model, tokenizer = FastEmbeddingModel.from_pretrained(
        m_cfg["base_model"],
        max_seq_length=m_cfg.get("max_seq_length", 512),
        pooling_strategy=m_cfg.get("pooling_strategy", "mean"),
    )
    model = FastEmbeddingModel.get_peft_model(
        model,
        r=int(m_cfg["lora"]["rank"]),
        lora_alpha=int(m_cfg["lora"]["alpha"]),
        lora_dropout=float(m_cfg["lora"]["dropout"]),
    )

    raw = json.loads(Path(train_data_path).read_text())
    train_ds = Dataset.from_list(raw)
    eval_ds = Dataset.from_list(json.loads(Path(val_data_path).read_text())) if val_data_path and Path(val_data_path).exists() else None

    negative_col = t_cfg.get("negative_column") or None

    args = EmbeddingSFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=int(t_cfg.get("batch_size", 32)),
        learning_rate=float(t_cfg["learning_rate"]),
        max_steps=int(t_cfg["iters"]),
        loss_type=t_cfg.get("loss_type", "infonce"),
        temperature=float(t_cfg.get("temperature", 0.05)),
        margin=float(t_cfg.get("margin", 1.0)),
        normalize_embeddings=bool(t_cfg.get("normalize_embeddings", True)),
        anchor_column=t_cfg.get("anchor_column", "anchor"),
        positive_column=t_cfg.get("positive_column", "positive"),
        negative_column=negative_col,
        max_seq_length=m_cfg.get("max_seq_length", 512),
    )

    trainer = EmbeddingSFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=args,
    )
    trainer.train()
