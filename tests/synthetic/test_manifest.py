import json
from data.synthetic.io import make_record
from data.synthetic.manifest import build_manifest, write_manifest

CFG = {"teacher": {"base_url": "http://x/v1", "model": "qwen3.6", "api_key": "secret"}}

def test_build_manifest_captures_teacher_and_counts_without_api_key():
    seeds = [make_record("c", "r", {})]
    m = build_manifest(CFG, seeds, {"generated": 100, "filtered": 60}, [4, 5, 5])
    assert m["teacher"] == {"base_url": "http://x/v1", "model": "qwen3.6"}
    assert "api_key" not in json.dumps(m)
    assert m["stage_counts"]["filtered"] == 60
    assert m["judge_score_distribution"] == {"4": 1, "5": 2}
    assert len(m["seed_set_hash"]) == 64

def test_write_manifest_creates_file(tmp_path):
    m = build_manifest(CFG, [], {}, [])
    write_manifest(tmp_path, m)
    assert json.loads((tmp_path / "manifest.json").read_text())["teacher"]["model"] == "qwen3.6"
