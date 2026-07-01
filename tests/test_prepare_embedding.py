import json
from pathlib import Path
from typer.testing import CliRunner
import sys

# Ensure the repo root is on the path for cli imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_domain(tmp_path: Path, domain_type: str = "embedding") -> Path:
    import yaml
    ws = tmp_path / "workspaces" / "emb"
    ws.mkdir(parents=True)
    (ws / "config.yaml").write_text(yaml.safe_dump({"type": domain_type}))
    return ws


def test_prepare_embedding_import_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ws = _make_domain(tmp_path)
    from cli import app
    runner = CliRunner()
    data_file = tmp_path / "pairs.json"
    data_file.write_text(json.dumps([
        {"anchor": "hello", "positive": "hi"},
        {"anchor": "bye", "positive": "goodbye"},
        {"anchor": "cat", "positive": "feline"},
        {"anchor": "dog", "positive": "canine"},
        {"anchor": "yes", "positive": "correct"},
    ]))
    result = runner.invoke(app, [
        "prepare-embedding", "emb",
        "--mode", "import",
        "--data-file", str(data_file),
        "--val-split", "0.2",
    ])
    assert result.exit_code == 0, result.output
    train = json.loads((ws / "processed" / "embedding_train.json").read_text())
    val = json.loads((ws / "processed" / "embedding_val.json").read_text())
    assert len(train) + len(val) == 5
    assert len(val) == 1  # 20% of 5 = 1
    assert all("anchor" in r and "positive" in r for r in train + val)


def test_prepare_embedding_import_jsonl(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ws = _make_domain(tmp_path)
    from cli import app
    runner = CliRunner()
    data_file = tmp_path / "pairs.jsonl"
    data_file.write_text(
        '\n'.join(json.dumps({"anchor": f"q{i}", "positive": f"a{i}"}) for i in range(5))
    )
    result = runner.invoke(app, [
        "prepare-embedding", "emb",
        "--mode", "import",
        "--data-file", str(data_file),
    ])
    assert result.exit_code == 0, result.output
    train = json.loads((ws / "processed" / "embedding_train.json").read_text())
    val = json.loads((ws / "processed" / "embedding_val.json").read_text())
    assert len(train) + len(val) == 5


def test_prepare_embedding_convert_mode(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ws = _make_domain(tmp_path)
    from cli import app
    runner = CliRunner()
    seeds_dir = ws / "seeds"
    seeds_dir.mkdir(parents=True)
    records = [
        {"conversation": [{"role": "user", "content": f"q{i}"},
                          {"role": "assistant", "content": f"a{i}"}]}
        for i in range(6)
    ]
    (seeds_dir / "approved.jsonl").write_text('\n'.join(json.dumps(r) for r in records))

    result = runner.invoke(app, ["prepare-embedding", "emb", "--mode", "convert"])
    assert result.exit_code == 0, result.output
    train = json.loads((ws / "processed" / "embedding_train.json").read_text())
    val = json.loads((ws / "processed" / "embedding_val.json").read_text())
    assert len(train) + len(val) == 6
    assert all("anchor" in r and "positive" in r for r in train + val)


def test_prepare_embedding_import_missing_column(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _make_domain(tmp_path)
    from cli import app
    runner = CliRunner()
    data_file = tmp_path / "bad.json"
    data_file.write_text(json.dumps([{"query": "hello", "doc": "hi"}]))
    result = runner.invoke(app, [
        "prepare-embedding", "emb",
        "--mode", "import",
        "--data-file", str(data_file),
    ])
    assert result.exit_code != 0
