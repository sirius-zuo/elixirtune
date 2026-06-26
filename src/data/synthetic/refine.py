from copy import deepcopy
from .teacher import Teacher

def _replace_assistant(record: dict, new_text: str, source: str) -> dict:
    out = deepcopy(record)
    out["conversation"][1]["content"] = new_text
    out["meta"]["source"] = source
    return out

def self_refine(record: dict, teacher: Teacher) -> dict:
    user = record["conversation"][0]["content"]
    assistant = record["conversation"][1]["content"]
    prompt = (
        f"INPUT:\n{user}\n\nDRAFT RESPONSE:\n{assistant}\n\n"
        "Improve the draft response. Return ONLY the improved response text."
    )
    return _replace_assistant(record, teacher.chat([{"role": "user", "content": prompt}]).strip(), "self_refine")

def critique_revise(record: dict, teacher: Teacher) -> dict:
    user = record["conversation"][0]["content"]
    assistant = record["conversation"][1]["content"]
    prompt = (
        f"INPUT:\n{user}\n\nRESPONSE:\n{assistant}\n\n"
        "First critique the response, then output the revised response after a line 'REVISED:'. "
        "Return the revised text only after that marker."
    )
    out = teacher.chat([{"role": "user", "content": prompt}])
    revised = out.split("REVISED:", 1)[1].strip() if "REVISED:" in out else out.strip()
    return _replace_assistant(record, revised, "critique_revise")

_PASSES = {"self_refine": self_refine, "critique_revise": critique_revise}

def apply_passes(record: dict, passes: list[str], teacher: Teacher) -> dict:
    out = record
    for name in passes:
        if name not in _PASSES:
            raise ValueError(f"unknown refinement pass: {name}")
        out = _PASSES[name](out, teacher)
    return out
