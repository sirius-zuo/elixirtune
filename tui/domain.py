from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import yaml


class Status(str, Enum):
    EMPTY = "empty"
    SEEDED = "seeded"
    GENERATED = "generated"
    PREPARED = "prepared"
    TRAINED = "trained"
    EVALUATED = "evaluated"
    DEPLOYED = "deployed"


@dataclass
class DomainState:
    name: str
    workspace: Path
    status: Status


def infer_status(ws: Path) -> Status:
    ws = Path(ws)
    if (ws / "fused").exists() and any((ws / "fused").iterdir()):
        return Status.DEPLOYED
    if (ws / "logs" / "evaluation").exists() and any(
        (ws / "logs" / "evaluation").glob("*_evaluation.json")
    ):
        return Status.EVALUATED
    if (ws / "adapters").exists() and any((ws / "adapters").iterdir()):
        return Status.TRAINED
    if (ws / "processed" / "train.json").exists():
        return Status.PREPARED
    if (ws / "generated" / "filtered.jsonl").exists():
        return Status.GENERATED
    if (ws / "seeds" / "approved.jsonl").exists():
        return Status.SEEDED
    return Status.EMPTY


def scan_domains(root: Path = Path(".")) -> list[DomainState]:
    workspaces = Path(root) / "workspaces"
    if not workspaces.exists():
        return []
    return [
        DomainState(name=ws.name, workspace=ws, status=infer_status(ws))
        for ws in sorted(workspaces.iterdir())
        if ws.is_dir()
    ]


def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out


def generate_runtime_configs(ws: Path, root: Path = Path(".")) -> None:
    ws, root = Path(ws), Path(root)
    ws.mkdir(parents=True, exist_ok=True)

    def overlay(base_file: str, overrides: dict, out_name: str) -> None:
        base_path = root / base_file
        base = yaml.safe_load(base_path.read_text()) if base_path.exists() else {}
        (ws / out_name).write_text(yaml.safe_dump(_deep_merge(base, overrides)))

    overlay("config/model_config.yaml", {
        "paths": {
            "adapter_dir": str(ws / "adapters"),
            "fused_model_dir": str(ws / "fused"),
            "checkpoint_dir": str(ws / "checkpoints"),
        }
    }, "runtime_model_config.yaml")

    overlay("config/training_config.yaml", {
        "paths": {
            "train_data": str(ws / "processed" / "train.json"),
            "test_data": str(ws / "processed" / "test.json"),
            "logs_dir": str(ws / "logs" / "training"),
        }
    }, "runtime_training_config.yaml")

    overlay("config/evaluation_config.yaml", {
        "paths": {
            "results_dir": str(ws / "logs" / "evaluation"),
            "test_data": str(ws / "processed" / "test.json"),
        }
    }, "runtime_eval_config.yaml")
