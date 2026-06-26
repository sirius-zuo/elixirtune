from data.synthetic.io import make_record, read_jsonl
from data.synthetic.assemble import strip_meta, assemble

def test_strip_meta_removes_meta_only():
    r = make_record("code", "review", {"source": "fewshot", "judge_score": 5})
    s = strip_meta(r)
    assert s == {"conversation": [
        {"role": "user", "content": "code"},
        {"role": "assistant", "content": "review"},
    ]}
    assert "meta" not in s

def test_assemble_writes_pure_contract(tmp_path):
    recs = [make_record("c", "r", {"source": "fewshot"})]
    out = tmp_path / "filtered.jsonl"
    assemble(recs, out)
    written = read_jsonl(out)
    assert written == [{"conversation": [
        {"role": "user", "content": "c"},
        {"role": "assistant", "content": "r"},
    ]}]
