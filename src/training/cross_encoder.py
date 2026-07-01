import json
import yaml
from pathlib import Path


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
    from sentence_transformers import CrossEncoder
    from sentence_transformers.cross_encoder.evaluation import CEBinaryAccuracyEvaluator
    from torch.utils.data import DataLoader
    from sentence_transformers import InputExample

    cfg = _load_configs(Path(model_config_path), Path(training_config_path))
    ce_cfg = cfg["model"].get("cross_encoder", {})
    t_cfg = cfg["training"]["embedding"]

    base_model = ce_cfg.get("base_model", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    max_seq_length = ce_cfg.get("max_seq_length", 512)
    output_dir = str(Path("workspaces") / domain / "ce_adapters")
    batch_size = int(t_cfg.get("batch_size", 16))
    num_epochs = max(1, int(t_cfg.get("iters", 100)) // 10)
    anchor_col = t_cfg.get("anchor_column", "anchor")
    positive_col = t_cfg.get("positive_column", "positive")
    negative_col = t_cfg.get("negative_column") or None

    records = json.loads(Path(train_data_path).read_text())

    # Build (query, doc, label) pairs: positives get 1.0, negatives get 0.0
    samples = []
    for rec in records:
        samples.append(InputExample(texts=[rec[anchor_col], rec[positive_col]], label=1.0))
        if negative_col and negative_col in rec and rec[negative_col]:
            samples.append(InputExample(texts=[rec[anchor_col], rec[negative_col]], label=0.0))

    model = CrossEncoder(base_model, max_length=max_seq_length, num_labels=1)
    train_dataloader = DataLoader(samples, shuffle=True, batch_size=batch_size)

    evaluator = None
    if val_data_path and Path(val_data_path).exists():
        val_records = json.loads(Path(val_data_path).read_text())
        val_pairs = [(r[anchor_col], r[positive_col]) for r in val_records]
        val_labels = [1] * len(val_pairs)
        if negative_col:
            for r in val_records:
                if negative_col in r and r[negative_col]:
                    val_pairs.append((r[anchor_col], r[negative_col]))
                    val_labels.append(0)
        evaluator = CEBinaryAccuracyEvaluator(
            sentence_pairs=val_pairs, labels=val_labels
        )

    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=num_epochs,
        output_path=output_dir,
    )
    print(f"Cross-encoder saved to {output_dir}")
