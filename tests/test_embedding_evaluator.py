import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np


def test_compute_similarity_returns_list_of_floats(tmp_path):
    with patch("mlx_tune.embeddings.FastEmbeddingModel") as MockModel:
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        MockModel.for_inference.return_value = mock_model
        mock_model.encode.side_effect = [
            np.array([[1.0, 0.0]]),
            np.array([[1.0, 0.0], [0.0, 1.0]]),
        ]
        import sys
        sys.path.insert(0, str(tmp_path))
        from src.evaluation.embedding_evaluator import compute_similarity
        scores = compute_similarity("hello", ["hi", "bye"], mock_model, mock_tokenizer)
        assert isinstance(scores, list)
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)


def test_recall_at_k_returns_dict(tmp_path):
    val_path = tmp_path / "embedding_val.json"
    records = [
        {"anchor": "q1", "positive": "a1"},
        {"anchor": "q2", "positive": "a2"},
    ]
    val_path.write_text(json.dumps(records))

    with patch("mlx_tune.embeddings.FastEmbeddingModel") as MockModel:
        mock_model = MagicMock()
        MockModel.for_inference.return_value = mock_model
        # encode returns (n, dim) for each call
        mock_model.encode.side_effect = [
            np.eye(2),   # anchor embeddings
            np.eye(2),   # positive embeddings
        ]
        import sys
        sys.path.insert(0, str(tmp_path))
        from src.evaluation.embedding_evaluator import recall_at_k
        result = recall_at_k(val_path, mock_model, MagicMock(), k=[1])
        assert "recall@1" in result
        assert 0.0 <= result["recall@1"] <= 1.0


def test_run_beir_graceful_without_package():
    # Since beir is not installed, the try/except will naturally trigger the fallback
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from src.evaluation.embedding_evaluator import run_beir
    result = run_beir("scifact", MagicMock(), MagicMock())
    assert result == {"error": "beir package not installed"}


def test_rerank_with_cross_encoder_returns_scores():
    with patch("sentence_transformers.CrossEncoder") as MockCE:
        mock_ce = MagicMock()
        MockCE.return_value = mock_ce
        mock_ce.predict.return_value = [0.9, 0.1, 0.5]
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.evaluation.embedding_evaluator import rerank_with_cross_encoder
        scores = rerank_with_cross_encoder("query", ["doc1", "doc2", "doc3"], "/fake/path")
        assert len(scores) == 3
        assert scores[0] == 0.9
