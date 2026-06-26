from pathlib import Path
import yaml
from data.synthetic.config import load_config

def test_domain_config_overrides_defaults(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "defaults.yaml").write_text(yaml.safe_dump({
        "teacher": {"model": "qwen3.6", "temperature": 0.8},
        "generate": {"target_size": 2000, "fewshot_k": 4},
    }))
    ws = tmp_path / "workspaces" / "code_review"
    ws.mkdir(parents=True)
    (ws / "config.yaml").write_text(yaml.safe_dump({
        "teacher": {"model": "prod-model"},
        "generate": {"target_size": 5000},
    }))

    cfg = load_config("code_review", root=tmp_path)

    assert cfg["teacher"]["model"] == "prod-model"      # overridden
    assert cfg["teacher"]["temperature"] == 0.8         # preserved from defaults
    assert cfg["generate"]["target_size"] == 5000       # overridden
    assert cfg["generate"]["fewshot_k"] == 4            # preserved from defaults

def test_missing_domain_config_uses_defaults_only(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "defaults.yaml").write_text(yaml.safe_dump({"generate": {"target_size": 2000}}))
    cfg = load_config("nonexistent", root=tmp_path)
    assert cfg["generate"]["target_size"] == 2000
