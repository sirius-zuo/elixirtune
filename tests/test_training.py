import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch


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


def test_dpo_trains_on_valid_preference_data(tmp_path):
    """DPOTrainer is built with the configured beta from {prompt,chosen,rejected} data."""
    pytest.importorskip("mlx_tune")  # training backend; skip when not installed
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    model_cfg = tmp_path / "model.yaml"
    model_cfg.write_text(
        "base_model:\n  path: m\nlora:\n  rank: 4\n  scale: 8.0\n  dropout: 0.0\n  keys: [q_proj]\n"
    )
    train_cfg = tmp_path / "train.yaml"
    train_cfg.write_text(
        "training:\n  batch_size: 2\n  learning_rate: 1e-5\n  iters: 10\n  steps_per_eval: 5\n"
        "dpo:\n  beta: 0.2\n"
    )
    train_data = tmp_path / "dpo.json"
    train_data.write_text(json.dumps([{"prompt": "p", "chosen": "c", "rejected": "r"}]))

    # Mock model load + trainer (no real training), but use the REAL DPOConfig so
    # the test actually exercises the mlx_tune API surface dpo.py depends on.
    with patch("mlx_tune.FastLanguageModel") as flm, \
         patch("mlx_tune.DPOTrainer") as trainer:
        flm.from_pretrained.return_value = (MagicMock(), MagicMock())
        flm.get_peft_model.return_value = MagicMock()
        from src.training.dpo import run
        run("d", model_cfg, train_cfg, train_data, None)

    _, kwargs = trainer.call_args
    assert kwargs["args"].beta == 0.2            # configured beta on a real DPOConfig
    assert "eval_dataset" not in kwargs          # DPOTrainer has no eval dataset
    trainer.return_value.train.assert_called_once()


def test_grpo_raises_without_correct_data(tmp_path):
    from src.training.grpo import run
    with pytest.raises(ValueError, match="GRPO requires"):
        run("d", tmp_path / "m.yaml", tmp_path / "t.yaml",
            tmp_path / "train.json", None)


def test_sft_does_not_pass_eval_steps_without_eval_dataset(tmp_path):
    """SFTConfig must not include eval_steps when val_data_path is None."""
    pytest.importorskip("mlx_tune")  # training backend; skip when not installed
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    model_cfg = tmp_path / "model.yaml"
    model_cfg.write_text(
        "base_model:\n  path: m\nlora:\n  rank: 4\n  scale: 8.0\n  dropout: 0.0\n  keys: [q_proj]\n"
    )
    train_cfg = tmp_path / "train.yaml"
    train_cfg.write_text(
        "training:\n  batch_size: 2\n  learning_rate: 1e-5\n  iters: 10\n  steps_per_eval: 5\n"
    )
    train_data = tmp_path / "train.json"
    train_data.write_text('[{"text": "hello"}]')

    captured_kwargs = {}

    def fake_sft_config(**kwargs):
        captured_kwargs.update(kwargs)
        return MagicMock()

    with patch("mlx_tune.FastLanguageModel") as mock_flm, \
         patch("mlx_tune.SFTTrainer") as mock_trainer, \
         patch("mlx_tune.SFTConfig", side_effect=fake_sft_config), \
         patch("src.training.sft.Dataset") as mock_dataset:
        mock_flm.from_pretrained.return_value = (MagicMock(), MagicMock())
        mock_flm.get_peft_model.return_value = MagicMock()
        mock_trainer.return_value.train.return_value = None
        mock_dataset.from_list.return_value = MagicMock()
        from src.training.sft import run
        run("d", model_cfg, train_cfg, train_data, None)

    assert "eval_steps" not in captured_kwargs, \
        f"eval_steps should not be passed when val_data is None, got {captured_kwargs}"
