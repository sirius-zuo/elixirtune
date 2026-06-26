from .io import write_jsonl

def strip_meta(record: dict) -> dict:
    return {"conversation": record["conversation"]}

def assemble(records: list[dict], out_path) -> list[dict]:
    stripped = [strip_meta(r) for r in records]
    write_jsonl(out_path, stripped)
    return stripped
