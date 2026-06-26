from copy import deepcopy
from pathlib import Path
import yaml

def _deep_merge(base: dict, override: dict) -> dict:
    out = deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = deepcopy(v)
    return out

def load_config(domain: str, root: Path = Path(".")) -> dict:
    root = Path(root)
    defaults = yaml.safe_load((root / "config" / "defaults.yaml").read_text()) or {}
    domain_path = root / "workspaces" / domain / "config.yaml"
    if domain_path.exists():
        override = yaml.safe_load(domain_path.read_text()) or {}
        return _deep_merge(defaults, override)
    return defaults
