import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path


def test_load_model_with_adapters_uses_public_api(tmp_path):
    """Ensure load_model_with_adapters calls mlx_lm.load with adapter_path kwarg."""
    adapter_dir = tmp_path / "adapters"
    adapter_dir.mkdir()

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()

    with patch("mlx_lm.load", return_value=(mock_model, mock_tokenizer)) as mock_load:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
        from evaluation.evaluator import ModelEvaluator

        eval_cfg = tmp_path / "eval.yaml"
        eval_cfg.write_text(
            "evaluation:\n  method: simple\n  max_tokens: 200\n  temperature: 0.7\n"
            "metrics:\n  simple:\n    word_overlap_threshold: 0.5\n"
            "comparison:\n  compare_with_base: true\n  score_thresholds:\n"
            "    excellent: 0.9\n    good: 0.7\n    acceptable: 0.5\n    poor: 0.3\n"
            "paths:\n  results_dir: /tmp\n  test_data: /tmp/test.json\n"
        )
        evaluator = ModelEvaluator(str(eval_cfg))
        model, tok = evaluator.load_model_with_adapters("base-model", str(adapter_dir))

    mock_load.assert_called_once_with("base-model", adapter_path=str(adapter_dir))
    assert model is mock_model
