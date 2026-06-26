import json
from data.synthetic.teacher import FakeTeacher
from data.synthetic.bootstrap import bootstrap_seeds


def test_bootstrap_parses_teacher_json_into_records():
    payload = json.dumps([
        {"user": "review this diff A", "assistant": "review A"},
        {"user": "review this diff B", "assistant": "review B"},
    ])
    t = FakeTeacher(responses=[payload])
    recs = bootstrap_seeds("code review domain", t, count=2)
    assert len(recs) == 2
    assert recs[0]["conversation"][0]["content"] == "review this diff A"
    assert recs[0]["meta"]["source"] == "bootstrap"


def test_bootstrap_skips_malformed_items():
    payload = json.dumps([{"user": "u", "assistant": "a"}, {"oops": 1}])
    t = FakeTeacher(responses=[payload])
    recs = bootstrap_seeds("desc", t, count=2)
    assert len(recs) == 1
