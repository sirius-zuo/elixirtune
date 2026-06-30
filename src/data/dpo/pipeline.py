"""Build DPO preference data: gather candidate answers per prompt from several
sources (teacher samples, the SFT model, the base model), judge-score them, and
pair the best as `chosen` and the worst as `rejected` when the score gap is wide
enough.

The orchestration here is backend-agnostic: candidate generation and judging are
injected, so the pipeline is fully testable without loading any model.
"""
import json
from pathlib import Path


def collect_prompts(domain: str, root: Path = Path("."), max_prompts: int = 200) -> list[str]:
    """User prompts to build preferences for, taken from the domain's curated
    seeds and generated data (deduped, capped)."""
    root = Path(root)
    ws = root / "workspaces" / domain
    prompts, seen = [], set()
    for f in (ws / "seeds" / "approved.jsonl", ws / "generated" / "filtered.jsonl"):
        if not f.exists():
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                conv = json.loads(line)["conversation"]
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            p = conv[0]["content"]
            if p not in seen:
                seen.add(p)
                prompts.append(p)
            if len(prompts) >= max_prompts:
                return prompts
    return prompts


def pair_candidates(prompt: str, candidates: list[str], judge_fn, min_margin: int) -> dict | None:
    """Score candidates and return a {prompt, chosen, rejected} pair, or None if
    there aren't 2 distinct candidates or the score gap is below min_margin."""
    uniq = list(dict.fromkeys(c for c in candidates if c and c.strip()))
    if len(uniq) < 2:
        return None
    scored = sorted(((judge_fn(prompt, c), c) for c in uniq), key=lambda x: x[0])
    (lo_score, rejected), (hi_score, chosen) = scored[0], scored[-1]
    if hi_score - lo_score < min_margin or chosen == rejected:
        return None
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def run_prepare_dpo(
    domain: str,
    gather_candidates,           # (prompts) -> list[list[str]] aligned with prompts
    judge_fn,                    # (prompt, response) -> int score
    min_margin: int = 2,
    max_prompts: int = 200,
    root: Path = Path("."),
    log=print,
) -> Path:
    prompts = collect_prompts(domain, root, max_prompts)
    if not prompts:
        raise ValueError(
            f"No prompts found for '{domain}'. Curate seeds (and optionally generate) first."
        )
    log(f"Collected {len(prompts)} prompts.")

    candidates = gather_candidates(prompts)

    pairs = []
    for prompt, cands in zip(prompts, candidates):
        pair = pair_candidates(prompt, cands, judge_fn, min_margin)
        if pair is not None:
            pairs.append(pair)
    log(f"Built {len(pairs)} preference pairs (margin >= {min_margin}) from {len(prompts)} prompts.")

    out = Path(root) / "workspaces" / domain / "processed" / "dpo.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(pairs, indent=2))
    log(f"Wrote {out}")
    return out
