import json
from typer.testing import CliRunner
from data.synthetic.io import read_jsonl
import cli

runner = CliRunner()


def test_init_with_seeds_imports_candidates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    seeds = tmp_path / "in.jsonl"
    seeds.write_text(json.dumps({"conversation": [
        {"role": "user", "content": "c"}, {"role": "assistant", "content": "r"}]}) + "\n")
    result = runner.invoke(cli.app, ["init", "code_review", "--seeds", str(seeds)])
    assert result.exit_code == 0
    assert len(read_jsonl(tmp_path / "workspaces" / "code_review" / "seeds" / "candidates.jsonl")) == 1


def test_curate_promotes_candidates_to_approved(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cand = tmp_path / "workspaces" / "code_review" / "seeds" / "candidates.jsonl"
    cand.parent.mkdir(parents=True)
    cand.write_text(json.dumps({"conversation": [
        {"role": "user", "content": "c"}, {"role": "assistant", "content": "r"}]}) + "\n")
    result = runner.invoke(cli.app, ["curate", "code_review"])
    assert result.exit_code == 0
    assert (cand.parent / "approved.jsonl").exists()


def test_prepare_out_dir_writes_to_custom_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ws = tmp_path / "workspaces" / "code_review"
    filtered = ws / "generated" / "filtered.jsonl"
    filtered.parent.mkdir(parents=True)
    filtered.write_text(json.dumps({"conversation": [
        {"role": "user", "content": "c"}, {"role": "assistant", "content": "r"}
    ]}) + "\n")
    out_dir = tmp_path / "custom_out"
    result = runner.invoke(cli.app, [
        "prepare", "code_review",
        "--system-prompt", "You are helpful.",
        "--out-dir", str(out_dir),
    ])
    assert result.exit_code == 0
    assert (out_dir / "train.json").exists()
