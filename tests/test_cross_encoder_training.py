import json
import yaml
from pathlib import Path
from unittest.mock import MagicMock, patch


def test_cross_encoder_run_creates_output_dir(tmp_path):
    model_cfg = tmp_path / "model_config.yaml"
    train_cfg = tmp_path / "training_config.yaml"
    train_data = tmp_path / "train.json"

    model_cfg.write_text(yaml.safe_dump({
        "cross_encoder": {
            "base_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "max_seq_length": 256,
        }
    }))
    train_cfg.write_text(yaml.safe_dump({
        "embedding": {
            "batch_size": 4,
            "learning_rate": 2e-5,
            "iters": 2,
            "anchor_column": "anchor",
            "positive_column": "positive",
            "negative_column": "negative",
        }
    }))
    train_data.write_text(json.dumps([
        {"anchor": "q1", "positive": "rel doc", "negative": "irrel doc"},
    ]))

    with patch("sentence_transformers.CrossEncoder") as MockCE:
        mock_ce = MagicMock()
        MockCE.return_value = mock_ce

        import sys
        sys.path.insert(0, str(tmp_path))
        from src.training.cross_encoder import run
        run("testdomain", model_cfg, train_cfg, train_data, None)

        mock_ce.fit.assert_called_once()


def test_cross_encoder_pairs_only_without_negatives(tmp_path):
    model_cfg = tmp_path / "model_config.yaml"
    train_cfg = tmp_path / "training_config.yaml"
    train_data = tmp_path / "train.json"

    model_cfg.write_text(yaml.safe_dump({
        "cross_encoder": {"base_model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "max_seq_length": 128}
    }))
    train_cfg.write_text(yaml.safe_dump({
        "embedding": {
            "batch_size": 4, "learning_rate": 2e-5, "iters": 2,
            "anchor_column": "anchor", "positive_column": "positive", "negative_column": None,
        }
    }))
    train_data.write_text(json.dumps([
        {"anchor": "q1", "positive": "doc1"},
        {"anchor": "q2", "positive": "doc2"},
    ]))

    with patch("sentence_transformers.CrossEncoder") as MockCE:
        mock_ce = MagicMock()
        MockCE.return_value = mock_ce

        import sys
        sys.path.insert(0, str(tmp_path))
        from src.training.cross_encoder import run
        run("testdomain", model_cfg, train_cfg, train_data, None)

        mock_ce.fit.assert_called_once()
