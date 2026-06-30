import json
from data.dpo.pipeline import collect_prompts, pair_candidates, run_prepare_dpo


def _seed(ws, prompts):
    (ws / "seeds").mkdir(parents=True, exist_ok=True)
    (ws / "seeds" / "approved.jsonl").write_text("\n".join(
        json.dumps({"conversation": [
            {"role": "user", "content": p}, {"role": "assistant", "content": "a"}], "meta": {}})
        for p in prompts))


def test_collect_prompts_dedups_and_caps(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    _seed(ws, ["p1", "p2", "p1", "p3"])
    assert collect_prompts("d", tmp_path) == ["p1", "p2", "p3"]
    assert collect_prompts("d", tmp_path, max_prompts=2) == ["p1", "p2"]


def test_pair_candidates_respects_margin():
    judge = lambda p, c: {"good": 5, "ok": 3, "bad": 1}[c]
    assert pair_candidates("p", ["good", "bad", "ok"], judge, min_margin=2) == \
        {"prompt": "p", "chosen": "good", "rejected": "bad"}
    # fewer than 2 distinct candidates → None
    assert pair_candidates("p", ["ok", "ok"], judge, 2) is None
    # margin below threshold → None
    judge2 = lambda p, c: {"a": 4, "b": 3}[c]
    assert pair_candidates("p", ["a", "b"], judge2, 2) is None


def test_run_prepare_dpo_writes_pairs(tmp_path):
    ws = tmp_path / "workspaces" / "d"
    _seed(ws, ["p1", "p2"])
    gather = lambda prompts: [["good", "bad"] for _ in prompts]
    judge = lambda p, c: 5 if c == "good" else 1
    out = run_prepare_dpo("d", gather, judge, min_margin=2, root=tmp_path, log=lambda m: None)
    data = json.loads(out.read_text())
    assert len(data) == 2
    assert all(set(d) == {"prompt", "chosen", "rejected"} for d in data)
    assert data[0] == {"prompt": "p1", "chosen": "good", "rejected": "bad"}
