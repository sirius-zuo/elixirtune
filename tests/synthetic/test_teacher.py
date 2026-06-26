import pytest
from data.synthetic.teacher import FakeTeacher, from_config, TeacherConfigError

def test_fake_teacher_returns_queued_responses_in_order():
    t = FakeTeacher(responses=["first", "second"])
    assert t.chat([{"role": "user", "content": "x"}]) == "first"
    assert t.chat([{"role": "user", "content": "y"}]) == "second"

def test_fake_teacher_records_calls():
    t = FakeTeacher(responses=["ok"])
    t.chat([{"role": "user", "content": "hello"}])
    assert t.calls[0][0]["content"] == "hello"

def test_from_config_rejects_empty_base_url():
    with pytest.raises(TeacherConfigError):
        from_config({"teacher": {"base_url": "", "model": "m", "api_key": ""}})
