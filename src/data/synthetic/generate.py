import json
import random
from .io import make_record
from .teacher import Teacher

class GenerationMiss(Exception):
    pass

def _fewshot_block(seeds: list[dict], k: int) -> str:
    chosen = random.sample(seeds, min(k, len(seeds)))
    lines = []
    for s in chosen:
        u = s["conversation"][0]["content"]
        a = s["conversation"][1]["content"]
        lines.append(f"INPUT:\n{u}\nRESPONSE:\n{a}")
    return "\n\n".join(lines)

def _parse_object(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise GenerationMiss("no JSON object in teacher output")
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise GenerationMiss(str(e))


def _parse_array(text: str) -> list:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise GenerationMiss("no JSON array in teacher output")
    try:
        data = json.loads(text[start:end + 1])
    except json.JSONDecodeError as e:
        raise GenerationMiss(str(e))
    if not isinstance(data, list):
        raise GenerationMiss("expected a JSON array")
    return data


def plan_topics(description: str, seeds: list[dict], teacher: Teacher,
                n: int, fewshot_k: int = 4, verbose: bool = False) -> list[str]:
    """Ask the teacher for a diverse list of sub-topics/scenarios for the domain.

    These steer each generation toward a different area, breaking the mode
    collapse that makes independent generations look near-identical.
    """
    prompt = (
        f"Domain: {description}\n\n"
        "Example tasks in this domain:\n\n"
        f"{_fewshot_block(seeds, fewshot_k)}\n\n"
        f"List {n} diverse, specific scenarios or sub-topics for this domain so that "
        "training examples built from them are varied — span different inputs, problem "
        "types, and focus areas. Return ONE scenario per line, no numbering, no commentary."
    )
    if verbose:
        print(f"--- topic plan request ---\n{prompt}", flush=True)
    raw = teacher.chat([{"role": "user", "content": prompt}])
    if verbose:
        print(f"--- topic plan response ---\n{raw}\n", flush=True)
    topics = []
    for line in raw.splitlines():
        s = line.strip().lstrip("-*•0123456789.) ").strip()
        if s:
            topics.append(s)
    return topics[:n]


def generate_batch(seeds: list[dict], teacher: Teacher, fewshot_k: int, cot: bool,
                   topics: list[str], batch_size: int, verbose: bool = False) -> list[dict]:
    """Generate a batch of DISTINCT pairs in one call, one per supplied topic.

    Asking for several distinct pairs at once (and steering each to a different
    topic) yields far more variety than repeated independent single generations.
    """
    cot_instr = (
        'For each pair, first reason step by step in "reasoning", then give the answer. '
        if cot else 'Leave "reasoning" empty. '
    )
    if topics:
        topic_block = (
            "Each pair must focus on a DIFFERENT one of these scenarios:\n"
            + "\n".join(f"- {t}" for t in topics) + "\n"
        )
        k = len(topics)
    else:
        topic_block = ""
        k = batch_size
    prompt = (
        "Here are example INPUT/RESPONSE pairs:\n\n"
        f"{_fewshot_block(seeds, fewshot_k)}\n\n"
        f"Produce {k} NEW, DISTINCT, realistic pairs in the same style. "
        f"{topic_block}"
        "Make every pair clearly different from each other and from the examples above. "
        f"{cot_instr}"
        'Return ONLY a JSON array of objects: '
        '[{"reasoning": "...", "user": "...", "assistant": "..."}, ...]'
    )
    if verbose:
        print("─" * 60, flush=True)
        print(f"--- request ({k} pairs) ---\n{prompt}", flush=True)
    raw = teacher.chat([{"role": "user", "content": prompt}])
    if verbose:
        print(f"--- response ---\n{raw}", flush=True)
        print("─" * 60, flush=True)
    records = []
    for i, obj in enumerate(_parse_array(raw)):
        if not isinstance(obj, dict) or "user" not in obj or "assistant" not in obj:
            continue
        topic = topics[i] if i < len(topics) else ""
        records.append(make_record(
            obj["user"], obj["assistant"],
            {"source": "fewshot-batch", "cot": obj.get("reasoning", ""), "topic": topic},
        ))
    if not records:
        raise GenerationMiss("no valid pairs in batch")
    return records

def generate_one(seeds: list[dict], teacher: Teacher, fewshot_k: int, cot: bool, verbose: bool = False) -> dict:
    cot_instr = (
        'First reason step by step in "reasoning", then give the final answer. '
        if cot else "Leave \"reasoning\" empty. "
    )
    prompt = (
        "Here are example INPUT/RESPONSE pairs:\n\n"
        f"{_fewshot_block(seeds, fewshot_k)}\n\n"
        "Produce ONE new, different, realistic pair in the same style. "
        f"{cot_instr}"
        'Return ONLY JSON: {"reasoning": "...", "user": "...", "assistant": "..."}'
    )
    if verbose:
        print("─" * 60, flush=True)
        print(f"--- request ---\n{prompt}", flush=True)
    raw = teacher.chat([{"role": "user", "content": prompt}])
    if verbose:
        print(f"--- response ---\n{raw}", flush=True)
        print("─" * 60, flush=True)
    obj = _parse_object(raw)
    if "user" not in obj or "assistant" not in obj:
        raise GenerationMiss("missing user/assistant")
    return make_record(obj["user"], obj["assistant"], {"source": "fewshot", "cot": obj.get("reasoning", "")})
