from data.synthetic.io import read_jsonl, write_jsonl, append_jsonl, make_record, sha256_of

def test_write_then_read_roundtrip(tmp_path):
    p = tmp_path / "x.jsonl"
    recs = [{"a": 1}, {"b": 2}]
    write_jsonl(p, recs)
    assert read_jsonl(p) == recs

def test_append_extends_existing(tmp_path):
    p = tmp_path / "x.jsonl"
    write_jsonl(p, [{"a": 1}])
    append_jsonl(p, [{"b": 2}])
    assert read_jsonl(p) == [{"a": 1}, {"b": 2}]

def test_read_missing_returns_empty(tmp_path):
    assert read_jsonl(tmp_path / "nope.jsonl") == []

def test_make_record_shape():
    r = make_record("code here", "review here", {"source": "fewshot"})
    assert r["conversation"] == [
        {"role": "user", "content": "code here"},
        {"role": "assistant", "content": "review here"},
    ]
    assert r["meta"]["source"] == "fewshot"

def test_sha256_is_stable_and_order_independent():
    assert sha256_of({"a": 1, "b": 2}) == sha256_of({"b": 2, "a": 1})
