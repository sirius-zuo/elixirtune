import json
import pytest
from pathlib import Path
from data.synthetic.io import make_record, write_jsonl, read_jsonl
from data.synthetic.teacher import FakeTeacher
from data.synthetic.embedder import FakeEmbedder
from data.synthetic.pipeline import run_generate, CurationGateError

def _cfg():
    return {
        "teacher": {"base_url": "http://x/v1", "model": "m", "api_key": ""},
        "generate": {"target_size": 2, "fewshot_k": 2, "cot": False},
        "refine": {"passes": []},
        "filter": {
            "schema": {"max_tokens": 2048},
            "dedup": {"similarity_threshold": 0.92},
            "judge": {"score_cutoff": 4},
            "diversity": {"quotas": {"bug": 1.0}},
        },
    }

def test_gate_blocks_without_approved_seeds(tmp_path):
    with pytest.raises(CurationGateError):
        run_generate("code_review", _cfg(), FakeTeacher(["x"]), FakeEmbedder({}), root=tmp_path)

def test_run_generate_produces_filtered_contract(tmp_path):
    ws = tmp_path / "workspaces" / "code_review"
    write_jsonl(ws / "seeds" / "approved.jsonl",
                [make_record("seed code", "seed review", {"source": "bootstrap"})])
    topics = "scenario alpha\nscenario beta"
    batch = json.dumps([
        {"reasoning": "", "user": "code 1", "assistant": "review one"},
        {"reasoning": "", "user": "code 2", "assistant": "review two"},
    ])
    # plan_topics, one batch (2 pairs), then two judge scores
    teacher = FakeTeacher(responses=[topics, batch, "5", "5"])
    embedder = FakeEmbedder({"review one": [1.0, 0.0], "review two": [0.0, 1.0]})
    run_dir = run_generate("code_review", _cfg(), teacher, embedder, root=tmp_path,
                           now="2026-06-25T14-30")
    assert (run_dir / "manifest.json").exists()
    filtered = read_jsonl(ws / "generated" / "filtered.jsonl")
    assert all("meta" not in r for r in filtered)
    assert all(set(r) == {"conversation"} for r in filtered)

def test_run_generate_resumes_from_existing_raw(tmp_path):
    ws = tmp_path / "workspaces" / "code_review"
    write_jsonl(ws / "seeds" / "approved.jsonl", [make_record("s", "r", {"source": "bootstrap"})])
    # Pre-seed one raw record so only one more is needed for target_size=2.
    write_jsonl(ws / "generated" / "raw.jsonl",
                [make_record("old", "old review", {"source": "fewshot", "category": "bug"})])
    topics = "scenario alpha\nscenario beta"
    batch = json.dumps([{"reasoning": "", "user": "n", "assistant": "fresh review"}])
    # plan_topics, one batch, then judge scores for the two kept records
    teacher = FakeTeacher(responses=[topics, batch, "5", "5"])
    embedder = FakeEmbedder({"old review": [1.0, 0.0], "fresh review": [0.0, 1.0]})
    run_generate("code_review", _cfg(), teacher, embedder, root=tmp_path, now="t1")
    raw = read_jsonl(ws / "generated" / "raw.jsonl")
    assert len(raw) == 2          # did not regenerate the first
