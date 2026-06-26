from datetime import datetime, timezone
from pathlib import Path
from . import generate as gen_mod
from .assemble import assemble
from .filter import validate_schema, dedup, judge, enforce_diversity
from .io import read_jsonl, append_jsonl, write_jsonl
from .manifest import build_manifest, write_manifest
from .refine import apply_passes

class CurationGateError(Exception):
    pass

def _categorize(record: dict) -> dict:
    # Lightweight default category until a richer classifier exists.
    record["meta"].setdefault("category", "bug")
    return record

def run_generate(domain, cfg, teacher, embedder, root=Path("."), now=None) -> Path:
    root = Path(root)
    ws = root / "workspaces" / domain
    seeds = read_jsonl(ws / "seeds" / "approved.jsonl")
    if not seeds:
        raise CurationGateError(f"no approved seeds for '{domain}' — run curate first")

    gcfg = cfg["generate"]
    raw_path = ws / "generated" / "raw.jsonl"
    have = len(read_jsonl(raw_path))                      # resume support
    misses = 0
    while have < gcfg["target_size"] and misses < gcfg["target_size"] * 5:
        try:
            rec = gen_mod.generate_one(seeds, teacher, gcfg["fewshot_k"], gcfg["cot"])
        except gen_mod.GenerationMiss:
            misses += 1
            continue
        append_jsonl(raw_path, [_categorize(rec)])
        have += 1

    raw = read_jsonl(raw_path)
    refined = [apply_passes(r, cfg["refine"]["passes"], teacher) for r in raw]
    write_jsonl(ws / "generated" / "refined.jsonl", refined)

    rejected = []
    kept, rej = validate_schema(refined, cfg["filter"]["schema"]["max_tokens"]); rejected += rej
    kept, rej = dedup(kept, embedder, cfg["filter"]["dedup"]["similarity_threshold"]); rejected += rej
    kept, rej = judge(kept, teacher, cfg["filter"]["judge"]["score_cutoff"]); rejected += rej
    kept, rej = enforce_diversity(kept, cfg["filter"]["diversity"]["quotas"], gcfg["target_size"]); rejected += rej

    assemble(kept, ws / "generated" / "filtered.jsonl")

    ts = now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M")
    run_dir = ws / "runs" / ts
    write_jsonl(run_dir / "rejected.jsonl", rejected)
    write_jsonl(run_dir / "stats.json", [{"kept": len(kept), "rejected": len(rejected)}])
    judge_scores = [r["meta"]["judge_score"] for r in kept if "judge_score" in r["meta"]]
    write_manifest(run_dir, build_manifest(cfg, seeds,
                   {"generated": len(raw), "filtered": len(kept)}, judge_scores))
    return run_dir
