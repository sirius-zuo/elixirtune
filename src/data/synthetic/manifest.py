import json
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from .io import sha256_of

def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"

def build_manifest(config: dict, seeds: list[dict], stage_counts: dict, judge_scores: list[int]) -> dict:
    t = config["teacher"]
    dist = {str(k): v for k, v in sorted(Counter(judge_scores).items())}
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "teacher": {"base_url": t["base_url"], "model": t["model"]},
        "seed_set_hash": sha256_of([s["conversation"] for s in seeds]),
        "stage_counts": stage_counts,
        "judge_score_distribution": dist,
        "git_sha": _git_sha(),
    }

def write_manifest(run_dir, manifest: dict) -> None:
    p = Path(run_dir)
    p.mkdir(parents=True, exist_ok=True)
    (p / "manifest.json").write_text(json.dumps(manifest, indent=2))
