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

def generate_one(seeds: list[dict], teacher: Teacher, fewshot_k: int, cot: bool) -> dict:
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
    obj = _parse_object(teacher.chat([{"role": "user", "content": prompt}]))
    if "user" not in obj or "assistant" not in obj:
        raise GenerationMiss("missing user/assistant")
    return make_record(obj["user"], obj["assistant"], {"source": "fewshot", "cot": obj.get("reasoning", "")})
