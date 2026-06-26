from data.synthetic.io import make_record
from data.synthetic.embedder import FakeEmbedder
from data.synthetic.teacher import FakeTeacher
from data.synthetic.filter import validate_schema, dedup, judge, enforce_diversity


def test_validate_schema_rejects_empty_and_too_long():
    good = make_record("code", "review", {})
    empty = make_record("code", "", {})
    longr = make_record("code", "x " * 5000, {})
    kept, rejected = validate_schema([good, empty, longr], max_tokens=2048)
    assert kept == [good]
    assert len(rejected) == 2
    assert all("reject_reason" in r["meta"] for r in rejected)


def test_dedup_drops_near_duplicates_above_threshold():
    a = make_record("c", "review one", {})
    b = make_record("c", "review one dup", {})   # near-duplicate
    c = make_record("c", "totally different", {})
    emb = FakeEmbedder({
        "review one": [1.0, 0.0], "review one dup": [0.99, 0.01],
        "totally different": [0.0, 1.0],
    })
    kept, rejected = dedup([a, b, c], emb, threshold=0.92)
    assert a in kept and c in kept
    assert len(rejected) == 1 and rejected[0]["conversation"][1]["content"] == "review one dup"
    assert "reject_reason" in rejected[0]["meta"]


def test_judge_keeps_at_or_above_cutoff():
    r1 = make_record("c", "good", {})
    r2 = make_record("c", "bad", {})
    t = FakeTeacher(responses=["5", "2"])
    kept, rejected = judge([r1, r2], t, score_cutoff=4)
    assert kept[0]["meta"]["judge_score"] == 5
    assert rejected[0]["meta"]["judge_score"] == 2


def test_enforce_diversity_respects_quota_and_target():
    bugs = [make_record("c", f"bug {i}", {"category": "bug"}) for i in range(10)]
    styles = [make_record("c", f"style {i}", {"category": "style"}) for i in range(10)]
    kept, rejected = enforce_diversity(bugs + styles, {"bug": 0.5, "style": 0.5}, target_size=4)
    assert sum(1 for r in kept if r["meta"]["category"] == "bug") == 2
    assert sum(1 for r in kept if r["meta"]["category"] == "style") == 2
