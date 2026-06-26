import json
import pytest
from data.synthetic.teacher import FakeTeacher
from data.synthetic.io import make_record
from data.synthetic.generate import generate_one, GenerationMiss

SEEDS = [make_record(f"code {i}", f"review {i}", {"source": "bootstrap"}) for i in range(5)]

def test_generate_one_strips_cot_from_assistant():
    payload = json.dumps({"reasoning": "think think", "user": "new code", "assistant": "clean review"})
    t = FakeTeacher(responses=[payload])
    rec = generate_one(SEEDS, t, fewshot_k=3, cot=True)
    assert rec["conversation"][1]["content"] == "clean review"
    assert rec["meta"]["cot"] == "think think"
    assert rec["meta"]["source"] == "fewshot"

def test_generate_one_raises_miss_on_malformed():
    t = FakeTeacher(responses=["not json at all"])
    with pytest.raises(GenerationMiss):
        generate_one(SEEDS, t, fewshot_k=3, cot=False)
