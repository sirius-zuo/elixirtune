import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock


def test_metrics_writer_callback_creates_file(tmp_path):
    from src.training.metrics_writer import MetricsWriterCallback
    cb = MetricsWriterCallback(tmp_path / "training_metrics.json")
    state = MagicMock(); state.global_step = 100
    cb.on_log(None, state, None, logs={"loss": 2.5})
    data = json.loads((tmp_path / "training_metrics.json").read_text())
    assert data["train_loss"] == [2.5]
    assert data["iterations"] == [100]
    assert data["val_loss"] == []


def test_metrics_writer_callback_appends_eval_loss(tmp_path):
    from src.training.metrics_writer import MetricsWriterCallback
    cb = MetricsWriterCallback(tmp_path / "training_metrics.json")
    state = MagicMock(); state.global_step = 50
    cb.on_log(None, state, None, logs={"loss": 2.0, "eval_loss": 2.1})
    data = json.loads((tmp_path / "training_metrics.json").read_text())
    assert data["val_loss"] == [2.1]


def test_dpo_raises_without_correct_data(tmp_path):
    from src.training.dpo import run
    with pytest.raises(ValueError, match="DPO requires"):
        run("d", tmp_path / "m.yaml", tmp_path / "t.yaml",
            tmp_path / "train.json", None)


def test_grpo_raises_without_correct_data(tmp_path):
    from src.training.grpo import run
    with pytest.raises(ValueError, match="GRPO requires"):
        run("d", tmp_path / "m.yaml", tmp_path / "t.yaml",
            tmp_path / "train.json", None)
