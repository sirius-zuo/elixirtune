from pathlib import Path
from tui.domain import Status, infer_status, scan_domains, generate_runtime_configs
import yaml

def test_infer_empty(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    ws.mkdir(parents=True)
    assert infer_status(ws) == Status.EMPTY

def test_infer_seeded(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "seeds").mkdir(parents=True)
    (ws / "seeds" / "approved.jsonl").write_text('{"conversation":[]}\n')
    assert infer_status(ws) == Status.SEEDED

def test_infer_generated(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "generated").mkdir(parents=True)
    (ws / "generated" / "filtered.jsonl").write_text('{"conversation":[]}\n')
    assert infer_status(ws) == Status.GENERATED

def test_infer_prepared(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "processed").mkdir(parents=True)
    (ws / "processed" / "train.json").write_text("[]")
    assert infer_status(ws) == Status.PREPARED

def test_infer_trained(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "adapters").mkdir(parents=True)
    (ws / "adapters" / "adapter.npz").write_text("x")
    assert infer_status(ws) == Status.TRAINED

def test_infer_evaluated(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "logs" / "evaluation").mkdir(parents=True)
    (ws / "logs" / "evaluation" / "base_model_evaluation.json").write_text("{}")
    assert infer_status(ws) == Status.EVALUATED

def test_infer_deployed(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "fused").mkdir(parents=True)
    (ws / "fused" / "weights.safetensors").write_text("x")
    assert infer_status(ws) == Status.DEPLOYED

def test_scan_domains_returns_sorted(tmp_path):
    for name in ["beta", "alpha"]:
        (tmp_path / "workspaces" / name).mkdir(parents=True)
    domains = scan_domains(tmp_path)
    assert [d.name for d in domains] == ["alpha", "beta"]

def test_scan_domains_empty_root(tmp_path):
    assert scan_domains(tmp_path) == []

def test_read_domain_type_defaults_to_lm(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    ws.mkdir(parents=True)
    from tui.domain import read_domain_type
    assert read_domain_type(ws) == "lm"


def test_read_domain_type_reads_config(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    ws.mkdir(parents=True)
    (ws / "config.yaml").write_text(yaml.safe_dump({"type": "embedding"}))
    from tui.domain import read_domain_type
    assert read_domain_type(ws) == "embedding"


def test_infer_status_embedding_empty(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    ws.mkdir(parents=True)
    (ws / "config.yaml").write_text(yaml.safe_dump({"type": "embedding"}))
    from tui.domain import infer_status, Status
    assert infer_status(ws) == Status.EMPTY


def test_infer_status_embedding_data_ready_seeds(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "seeds").mkdir(parents=True)
    (ws / "seeds" / "approved.jsonl").write_text("{}\n")
    (ws / "config.yaml").write_text(yaml.safe_dump({"type": "embedding"}))
    from tui.domain import infer_status, Status
    assert infer_status(ws) == Status.DATA_READY


def test_infer_status_embedding_data_ready_raw(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "data" / "raw").mkdir(parents=True)
    (ws / "data" / "raw" / "pairs.json").write_text("[]")
    (ws / "config.yaml").write_text(yaml.safe_dump({"type": "embedding"}))
    from tui.domain import infer_status, Status
    assert infer_status(ws) == Status.DATA_READY


def test_infer_status_embedding_prepared(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "processed").mkdir(parents=True)
    (ws / "processed" / "embedding_train.json").write_text("[]")
    (ws / "processed" / "embedding_val.json").write_text("[]")
    (ws / "config.yaml").write_text(yaml.safe_dump({"type": "embedding"}))
    from tui.domain import infer_status, Status
    assert infer_status(ws) == Status.PREPARED


def test_infer_status_embedding_trained(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "adapters").mkdir(parents=True)
    (ws / "adapters" / "adapter.npz").write_text("x")
    (ws / "config.yaml").write_text(yaml.safe_dump({"type": "embedding"}))
    from tui.domain import infer_status, Status
    assert infer_status(ws) == Status.TRAINED


def test_infer_status_embedding_ce_trained(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "adapters").mkdir(parents=True)
    (ws / "adapters" / "adapter.npz").write_text("x")
    (ws / "ce_adapters").mkdir(parents=True)
    (ws / "ce_adapters" / "pytorch_model.bin").write_text("x")
    (ws / "config.yaml").write_text(yaml.safe_dump({"type": "embedding"}))
    from tui.domain import infer_status, Status
    assert infer_status(ws) == Status.CE_TRAINED


def test_existing_lm_status_unaffected(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (ws / "processed").mkdir(parents=True)
    (ws / "processed" / "train.json").write_text("[]")
    assert infer_status(ws) == Status.PREPARED


def test_generate_runtime_configs_writes_overlays(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "model_config.yaml").write_text(
        yaml.safe_dump({"base_model": {"path": "Phi-3"}, "paths": {"adapter_dir": "models/adapters", "fused_model_dir": "models/fused", "checkpoint_dir": "models/checkpoints"}})
    )
    (tmp_path / "config" / "training_config.yaml").write_text(
        yaml.safe_dump({"training": {"iters": 2000}, "paths": {"train_data": "data/processed/train.json", "test_data": "data/processed/test.json", "logs_dir": "logs/training"}})
    )
    (tmp_path / "config" / "evaluation_config.yaml").write_text(
        yaml.safe_dump({"evaluation": {"method": "simple"}, "paths": {"results_dir": "logs/evaluation", "test_data": "data/processed/test.json"}})
    )
    generate_runtime_configs(ws, root=tmp_path)
    model_cfg = yaml.safe_load((ws / "runtime_model_config.yaml").read_text())
    assert model_cfg["paths"]["adapter_dir"] == str(ws / "adapters")
    assert model_cfg["paths"]["fused_model_dir"] == str(ws / "fused")
    assert model_cfg["base_model"]["path"] == "Phi-3"  # base preserved
    train_cfg = yaml.safe_load((ws / "runtime_training_config.yaml").read_text())
    assert train_cfg["paths"]["train_data"] == str(ws / "processed" / "train.json")
    assert train_cfg["paths"]["logs_dir"] == str(ws / "logs" / "training")
    assert train_cfg["training"]["iters"] == 2000  # base preserved
    eval_cfg = yaml.safe_load((ws / "runtime_eval_config.yaml").read_text())
    assert eval_cfg["paths"]["results_dir"] == str(ws / "logs" / "evaluation")
    assert eval_cfg["paths"]["test_data"] == str(ws / "processed" / "test.json")
