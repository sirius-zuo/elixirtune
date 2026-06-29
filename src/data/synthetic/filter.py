from copy import deepcopy
from .embedder import Embedder
from .teacher import Teacher


def _reject(record: dict, reason: str) -> dict:
    out = deepcopy(record)
    out["meta"]["reject_reason"] = reason
    return out


def _assistant(record: dict) -> str:
    return record["conversation"][1]["content"]


def validate_schema(records, max_tokens: int):
    kept, rejected = [], []
    for r in records:
        a = _assistant(r).strip()
        if not a:
            rejected.append(_reject(r, "empty_assistant"))
        elif len(a.split()) > max_tokens:
            rejected.append(_reject(r, "too_long"))
        else:
            kept.append(r)
    return kept, rejected


def _cosine(u, v):
    dot = sum(x * y for x, y in zip(u, v))
    nu = sum(x * x for x in u) ** 0.5
    nv = sum(y * y for y in v) ** 0.5
    return dot / (nu * nv) if nu and nv else 0.0


def dedup(records, embedder: Embedder, threshold: float):
    vecs = embedder.embed([_assistant(r) for r in records])
    kept, rejected, kept_vecs = [], [], []
    for r, v in zip(records, vecs):
        if any(_cosine(v, kv) >= threshold for kv in kept_vecs):
            rejected.append(_reject(r, "near_duplicate"))
        else:
            kept.append(r)
            kept_vecs.append(v)
    return kept, rejected


def judge(records, teacher: Teacher, score_cutoff: int, verbose: bool = False):
    kept, rejected = [], []
    for r in records:
        prompt = (
            f"INPUT:\n{r['conversation'][0]['content']}\n\nRESPONSE:\n{_assistant(r)}\n\n"
            "Rate the response quality from 1 to 5. Return ONLY the integer."
        )
        try:
            score = int("".join(c for c in teacher.chat([{"role": "user", "content": prompt}]) if c.isdigit())[:1])
        except (ValueError, IndexError):
            score = 0
        out = deepcopy(r)
        out["meta"]["judge_score"] = score
        passed = score >= score_cutoff
        if verbose:
            preview = " ".join(r["conversation"][0]["content"].split())[:60]
            print(f"  judge score={score} {'kept' if passed else 'rejected'} | {preview}", flush=True)
        if passed:
            kept.append(out)
        else:
            out["meta"]["reject_reason"] = "below_judge_cutoff"
            rejected.append(out)
    return kept, rejected


def enforce_diversity(records, quotas: dict[str, float], target_size: int):
    caps = {cat: int(round(frac * target_size)) for cat, frac in quotas.items()}
    counts = {cat: 0 for cat in quotas}
    kept, rejected = [], []
    for r in records:
        cat = r["meta"].get("category")
        if cat in caps and counts[cat] < caps[cat]:
            counts[cat] += 1
            kept.append(r)
        else:
            rejected.append(_reject(r, "diversity_quota_full"))
    return kept, rejected
