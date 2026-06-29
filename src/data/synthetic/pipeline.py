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

def run_generate(domain, cfg, teacher, embedder, root=Path("."), now=None, verbose=False) -> Path:
    def log(msg):
        print(msg, flush=True)

    root = Path(root)
    ws = root / "workspaces" / domain
    seeds = read_jsonl(ws / "seeds" / "approved.jsonl")
    if not seeds:
        raise CurationGateError(f"no approved seeds for '{domain}' — run curate first")

    gcfg = cfg["generate"]
    target = gcfg["target_size"]
    raw_path = ws / "generated" / "raw.jsonl"
    have = len(read_jsonl(raw_path))                      # resume support
    log(f"Generating up to {target} examples for '{domain}' (have {have})…")
    misses = 0
    while have < target and misses < target * gcfg.get("max_miss_factor", 5):
        try:
            rec = gen_mod.generate_one(seeds, teacher, gcfg["fewshot_k"], gcfg["cot"], verbose=verbose)
        except gen_mod.GenerationMiss as e:
            misses += 1
            if verbose:
                log(f"  miss ({misses}): {e}")
            continue
        append_jsonl(raw_path, [_categorize(rec)])
        have += 1
        log(f"[gen] {have}/{target}  (misses {misses})")

    raw = read_jsonl(raw_path)
    log(f"Refining {len(raw)} examples…")
    refined = [apply_passes(r, cfg["refine"]["passes"], teacher) for r in raw]
    write_jsonl(ws / "generated" / "refined.jsonl", refined)

    log("Filtering:")
    rejected = []
    kept, rej = validate_schema(refined, cfg["filter"]["schema"]["max_tokens"]); rejected += rej
    log(f"  schema:    {len(kept)} kept, {len(rej)} rejected")
    kept, rej = dedup(kept, embedder, cfg["filter"]["dedup"]["similarity_threshold"]); rejected += rej
    log(f"  dedup:     {len(kept)} kept, {len(rej)} rejected")
    cutoff = cfg["filter"]["judge"]["score_cutoff"]
    kept, rej = judge(kept, teacher, cutoff, verbose=verbose); rejected += rej
    log(f"  judge:     {len(kept)} kept, {len(rej)} rejected (cutoff {cutoff})")
    kept, rej = enforce_diversity(kept, cfg["filter"]["diversity"]["quotas"], target); rejected += rej
    log(f"  diversity: {len(kept)} kept, {len(rej)} rejected")

    assemble(kept, ws / "generated" / "filtered.jsonl")
    log(f"Done: {len(kept)} examples → generated/filtered.jsonl")

    ts = now or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M")
    run_dir = ws / "runs" / ts
    write_jsonl(run_dir / "rejected.jsonl", rejected)
    write_jsonl(run_dir / "stats.json", [{"kept": len(kept), "rejected": len(rejected)}])
    judge_scores = [r["meta"]["judge_score"] for r in kept if "judge_score" in r["meta"]]
    write_manifest(run_dir, build_manifest(cfg, seeds,
                   {"generated": len(raw), "filtered": len(kept)}, judge_scores))
    return run_dir
