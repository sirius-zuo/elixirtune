import json
from pathlib import Path

import yaml


def _load_configs(model_config_path: Path, training_config_path: Path) -> dict:
    m = yaml.safe_load(Path(model_config_path).read_text())
    t = yaml.safe_load(Path(training_config_path).read_text())
    return {"model": m, "training": t}


def run(
    domain: str,
    model_config_path: Path,
    training_config_path: Path,
    train_data_path: Path,
    val_data_path: Path | None,   # unused: mlx_tune's DPOTrainer has no eval dataset
) -> None:
    # Validate the preference-data contract before importing the heavy backend,
    # so a misconfigured run fails fast with a clear message.
    data = json.loads(Path(train_data_path).read_text()) if Path(train_data_path).exists() else []
    if not data or not isinstance(data[0], dict) or "chosen" not in data[0]:
        raise ValueError(
            "DPO requires training data with fields {prompt, chosen, rejected}. "
            "Run the DPO data preparation pipeline first."
        )

    import src._compat  # noqa: F401 — apply Python 3.14 / datasets compat patches
    from datasets import Dataset
    from mlx_tune import FastLanguageModel, DPOTrainer, DPOConfig

    cfg = _load_configs(Path(model_config_path), Path(training_config_path))
    m_cfg = cfg["model"]
    train_cfg = cfg["training"]["training"]
    dpo_cfg = cfg["training"].get("dpo", {})

    output_dir = str(Path("workspaces") / domain / "adapters")

    # DPO trains a fresh LoRA on the base model; DPOTrainer builds the frozen
    # reference model internally (ref_model=None).
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

    train_ds = Dataset.from_list(json.loads(Path(train_data_path).read_text()))

    # mlx_tune's DPOTrainer is a native trainer (no transformers callbacks); it
    # streams "Step n/iters | Loss: ..." to stdout, which the TUI shows in the log.
    args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=int(train_cfg["batch_size"]),
        learning_rate=float(train_cfg["learning_rate"]),
        max_steps=int(train_cfg["iters"]),
        beta=float(dpo_cfg.get("beta", 0.1)),
    )
    trainer = DPOTrainer(
        model=model,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        args=args,
    )
    trainer.train()
