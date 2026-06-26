import pytest
from data.synthetic.teacher import FakeTeacher
from data.synthetic.io import make_record
from data.synthetic.refine import self_refine, apply_passes

BASE = make_record("code", "ok review", {"source": "fewshot", "cot": ""})

def test_self_refine_replaces_assistant_and_tags_source():
    t = FakeTeacher(responses=["better review"])
    out = self_refine(BASE, t)
    assert out["conversation"][1]["content"] == "better review"
    assert out["meta"]["source"] == "self_refine"
    assert out["conversation"][0]["content"] == "code"  # user untouched

def test_apply_passes_empty_is_noop():
    t = FakeTeacher(responses=["x"])
    assert apply_passes(BASE, [], t) == BASE

def test_apply_passes_runs_in_order():
    t = FakeTeacher(responses=["after_self", "after_critique"])
    out = apply_passes(BASE, ["self_refine", "critique_revise"], t)
    assert out["conversation"][1]["content"] == "after_critique"
    assert out["meta"]["source"] == "critique_revise"

def test_apply_passes_unknown_raises():
    with pytest.raises(ValueError):
        apply_passes(BASE, ["nope"], FakeTeacher(responses=["x"]))
