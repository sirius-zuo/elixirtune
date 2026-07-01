import json
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch


def _write_config(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data))


def test_embedding_run_loads_correct_config(tmp_path):
    """run() reads embedding block from model config, not top-level lm keys."""
    model_cfg = tmp_path / "model_config.yaml"
    train_cfg = tmp_path / "training_config.yaml"
    train_data = tmp_path / "train.json"
    _write_config(model_cfg, {
        "embedding": {
            "base_model": "mlx-community/all-MiniLM-L6-v2",
            "max_seq_length": 128,
            "pooling_strategy": "mean",
            "lora": {"rank": 8, "alpha": 8, "dropout": 0.0},
        }
    })
    _write_config(train_cfg, {
        "embedding": {
            "batch_size": 4,
            "learning_rate": 2e-5,
            "iters": 2,
            "loss_type": "infonce",
            "temperature": 0.05,
            "margin": 1.0,
            "normalize_embeddings": True,
            "anchor_column": "anchor",
            "positive_column": "positive",
            "negative_column": None,
        }
    })
    _write_data = tmp_path / "data.json"
    _write_data.write_text(json.dumps([
        {"anchor": "hello", "positive": "hi there"},
        {"anchor": "goodbye", "positive": "see you"},
    ]))

    with patch("mlx_tune.embeddings.FastEmbeddingModel") as MockModel, \
         patch("mlx_tune.embeddings.EmbeddingSFTTrainer") as MockTrainer, \
         patch("mlx_tune.embeddings.EmbeddingSFTConfig") as MockConfig:
        mock_model = MagicMock()
        MockModel.from_pretrained.return_value = (mock_model, MagicMock())
        MockModel.get_peft_model.return_value = mock_model

        import sys
        sys.path.insert(0, str(tmp_path))
        from src.training.embedding import run
        run("testdomain", model_cfg, train_cfg, _write_data, None)

        MockModel.from_pretrained.assert_called_once()
        call_kwargs = MockModel.from_pretrained.call_args
        assert call_kwargs[0][0] == "mlx-community/all-MiniLM-L6-v2"
        MockTrainer.return_value.train.assert_called_once()
