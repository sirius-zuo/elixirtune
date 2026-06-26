import json
from .io import make_record
from .teacher import Teacher

_PROMPT = (
    "You are creating starter training examples for this domain:\n\n{desc}\n\n"
    "Generate {count} diverse, realistic examples. Return ONLY a JSON array where each "
    'item is {{"user": "<task input>", "assistant": "<ideal response>"}}.'
)


def _parse_array(text: str) -> list[dict]:
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return []


def bootstrap_seeds(domain_desc: str, teacher: Teacher, count: int) -> list[dict]:
    text = teacher.chat([{"role": "user", "content": _PROMPT.format(desc=domain_desc, count=count)}])
    records = []
    for item in _parse_array(text):
        if isinstance(item, dict) and "user" in item and "assistant" in item:
            records.append(make_record(item["user"], item["assistant"], {"source": "bootstrap"}))
    return records
