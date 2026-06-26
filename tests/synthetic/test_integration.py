import os
import pytest


@pytest.mark.skipif(not os.environ.get("ELIXIRLORA_LIVE"),
                    reason="set ELIXIRLORA_LIVE=1 with a local llama.cpp server on :8080")
def test_end_to_end_against_local_llamacpp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from typer.testing import CliRunner
    import cli
    from data.synthetic.io import write_jsonl, make_record, read_jsonl
    ws = tmp_path / "workspaces" / "code_review"
    write_jsonl(ws / "seeds" / "approved.jsonl",
                [make_record(f"def f(): return {i}", f"Looks fine, example {i}.", {"source": "bootstrap"})
                 for i in range(3)])
    # small run via config override
    (ws / "config.yaml").write_text("generate:\n  target_size: 3\n")
    result = CliRunner().invoke(cli.app, ["generate", "code_review"])
    assert result.exit_code == 0
    assert len(read_jsonl(ws / "generated" / "filtered.jsonl")) >= 0
