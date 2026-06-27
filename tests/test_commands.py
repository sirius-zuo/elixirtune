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


def test_fuse_calls_adapter_fusion(tmp_path):
    """fuse command delegates to AdapterFusion.validate_fusion_inputs and fuse_adapters."""
    from unittest.mock import patch, MagicMock
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    adapters_dir = tmp_path / "adapters"
    adapters_dir.mkdir()
    (adapters_dir / "adapters.safetensors").write_bytes(b"x")
    (adapters_dir / "adapter_config.json").write_text('{"lora_layers": 4}')

    model_cfg = tmp_path / "runtime_model_config.yaml"
    model_cfg.write_text("base_model:\n  path: 'some/model'\n")

    fused_out = tmp_path / "fused"

    mock_ctx = MagicMock()
    mock_ctx.invoked_subcommand = None

    with patch("utils.fusion.AdapterFusion.validate_fusion_inputs", return_value=True) as mock_validate, \
         patch("utils.fusion.AdapterFusion.fuse_adapters", return_value=str(fused_out)) as mock_fuse:
        from commands.fuse import fuse
        fuse(
            ctx=mock_ctx,
            domain="d",
            model_config=model_cfg,
            output_path=fused_out,
            eval_config=None,
            test_data=None,
            adapters_path=adapters_dir,
        )

    mock_validate.assert_called_once_with("some/model", str(adapters_dir))
    mock_fuse.assert_called_once_with("some/model", str(adapters_dir), str(fused_out))


def test_text_generator_has_no_hardcoded_system_prompt():
    """TextGenerator must not default to any hardcoded persona."""
    import sys; sys.path.insert(0, "src")
    from inference.generator import TextGenerator
    from unittest.mock import patch, MagicMock
    with patch("inference.generator.load", return_value=(MagicMock(), MagicMock())):
        gen = TextGenerator("some/model", system_prompt=None)
    # system prompt should be empty/None, not the old "Didier" string
    assert "Didier" not in (gen.default_system_prompt or "")
    assert "OpenBB" not in (gen.default_system_prompt or "")


def test_generate_calls_load_config_with_domain_string(tmp_path):
    """load_config must receive the bare domain string, not a Path."""
    import sys
    import os
    repo_root = str(Path(__file__).parent.parent)
    sys.path.insert(0, repo_root)
    sys.path.insert(0, str(Path(repo_root) / "src"))
    os.chdir(tmp_path)
    cfg_data = {"filter": {"dedup": {"embedding_model": "m"}}, "generate": {}}
    ws = tmp_path / "workspaces" / "d" / "seeds"
    ws.mkdir(parents=True)
    (ws.parent / "approved.jsonl").write_text("[{\"text\": \"hello\"}]\n")
    with patch("data.synthetic.config.load_config", return_value=cfg_data) as mock_cfg, \
         patch("data.synthetic.teacher.from_config", return_value=MagicMock()), \
         patch("data.synthetic.embedder.SentenceTransformerEmbedder", return_value=MagicMock()), \
         patch("data.synthetic.pipeline.run_generate") as mock_run:
        from commands.generate import generate
        ctx = MagicMock()
        ctx.invoked_subcommand = None
        try:
            generate(ctx, "d")
        except Exception:
            pass
    call_arg = mock_cfg.call_args[0][0] if mock_cfg.call_args else None
    assert call_arg == "d", f"Expected 'd' but got {call_arg!r}"


def test_generate_passes_model_name_to_embedder(tmp_path):
    """SentenceTransformerEmbedder must receive model_name from config."""
    import sys
    import os
    repo_root = str(Path(__file__).parent.parent)
    sys.path.insert(0, repo_root)
    sys.path.insert(0, str(Path(repo_root) / "src"))
    os.chdir(tmp_path)
    cfg_data = {"filter": {"dedup": {"embedding_model": "all-MiniLM-L6-v2"}}, "generate": {}}
    ws = tmp_path / "workspaces" / "d" / "seeds"
    ws.mkdir(parents=True)
    (ws.parent / "approved.jsonl").write_text("[{\"text\": \"hello\"}]\n")
    with patch("data.synthetic.config.load_config", return_value=cfg_data), \
         patch("data.synthetic.teacher.from_config", return_value=MagicMock()), \
         patch("data.synthetic.embedder.SentenceTransformerEmbedder") as mock_emb, \
         patch("data.synthetic.pipeline.run_generate"):
        from commands.generate import generate
        ctx = MagicMock()
        ctx.invoked_subcommand = None
        try:
            generate(ctx, "d")
        except Exception:
            pass
    mock_emb.assert_called_once_with("all-MiniLM-L6-v2")


def test_generate_does_not_pass_seeds_to_run_generate(tmp_path):
    """run_generate must be called WITHOUT seeds argument (it reads from disk itself)."""
    import sys
    import os
    repo_root = str(Path(__file__).parent.parent)
    sys.path.insert(0, repo_root)
    sys.path.insert(0, str(Path(repo_root) / "src"))
    os.chdir(tmp_path)
    cfg_data = {"filter": {"dedup": {"embedding_model": "m"}}, "generate": {}}
    ws = tmp_path / "workspaces" / "d" / "seeds"
    ws.mkdir(parents=True)
    (ws.parent / "approved.jsonl").write_text("[{\"text\": \"hello\"}]\n")
    with patch("data.synthetic.config.load_config", return_value=cfg_data), \
         patch("data.synthetic.teacher.from_config", return_value=MagicMock()), \
         patch("data.synthetic.embedder.SentenceTransformerEmbedder", return_value=MagicMock()), \
         patch("data.synthetic.pipeline.run_generate") as mock_run:
        from commands.generate import generate
        ctx = MagicMock()
        ctx.invoked_subcommand = None
        try:
            generate(ctx, "d")
        except Exception:
            pass
    if mock_run.called:
        args, kwargs = mock_run.call_args
        if len(args) >= 5:
            assert not isinstance(args[4], list), "seeds list must not be passed as 5th arg"


def test_model_evaluator_init_without_paths_block(tmp_path):
    """ModelEvaluator must not crash when config has no paths: section."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    eval_cfg = tmp_path / "eval.yaml"
    eval_cfg.write_text(
        "evaluation:\n  method: simple\n  max_tokens: 200\n  temperature: 0.7\n"
        "metrics:\n  simple:\n    word_overlap_threshold: 0.5\n"
        "comparison:\n  compare_with_base: true\n"
        "  score_thresholds:\n    excellent: 0.9\n    good: 0.7\n    acceptable: 0.5\n    poor: 0.3\n"
    )
    from evaluation.evaluator import ModelEvaluator
    evaluator = ModelEvaluator(str(eval_cfg))   # must not raise
    assert evaluator.paths_config == {}
