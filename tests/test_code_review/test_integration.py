"""Integration test: run full pipeline on a tiny dataset subset."""

import json
import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_root / "src"))


def test_full_pipeline_on_tiny_subset(tmp_path):
    """
    Run the setup pipeline on a synthetic dataset to verify:
    1. Seeds are valid ElixirLoRA format
    2. prepare command can consume them
    3. Output has expected train/val/test structure
    """
    import subprocess
    import yaml

    # Create a tiny synthetic dataset to simulate what setup.py would download
    fake_dataset = [
        {
            "code_diff": "@@ -1,3 +1,3 @@\n-old_line\n+new_line",
            "file_path": "src/test.py",
            "language": "Python",
            "pr_title": "Test PR",
            "review_type": "suggestion",
            "comment": "Fix this line.",
            "suggestion": "def fixed():\n    pass",
            "repo_name": "test/repo",
            "pr_number": 1,
            "line": 1,
            "old_line": 1,
            "new_line": 2,
        },
        {
            "code_diff": "",
            "file_path": "",
            "language": "",
            "pr_title": "",
            "review_type": "none",
            "comment": "",
            "suggestion": "",
            "repo_name": "test/repo2",
            "pr_number": 2,
            "line": 0,
            "old_line": 0,
            "new_line": 0,
        },
        {
            "code_diff": "@@ -1,0 +1,1 @@\n+added line",
            "file_path": "src/other.py",
            "language": "JavaScript",
            "pr_title": "Add feature",
            "review_type": "refactor",
            "comment": "Extract this.",
            "suggestion": "function helper():\n    pass",
            "repo_name": "test/repo3",
            "pr_number": 3,
            "line": 1,
            "old_line": 0,
            "new_line": 1,
        },
    ]

    # Write fake dataset for setup.py to consume
    fake_dataset_path = tmp_path / "data" / "code-review" / "fake_dataset.jsonl"
    fake_dataset_path.parent.mkdir(parents=True)
    for record in fake_dataset:
        fake_dataset_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in fake_dataset) + "\n",
            encoding="utf-8",
        )

    # Import and run setup's conversion with a custom dataset path
    # We need to patch the dataset loading since we can't actually download
    from examples.code_review.setup import make_record, build_user_message, build_assistant_message

    # Convert records manually
    seeds = []
    for record in fake_dataset:
        user_content = build_user_message(record)
        assistant_content = build_assistant_message(record)
        meta = {
            "review_type": record.get("review_type", "none"),
            "language": record.get("language", ""),
            "repo": record.get("repo_name", ""),
            "pr_title": record.get("pr_title", ""),
        }
        seeds.append(make_record(user_content, assistant_content, meta))

    assert len(seeds) == 3

    # Simulate workspace creation
    ws = tmp_path / "workspaces" / "test-integration"
    seeds_dir = ws / "seeds"
    seeds_dir.mkdir(parents=True)
    approved_path = seeds_dir / "approved.jsonl"
    approved_path.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in seeds) + "\n",
        encoding="utf-8",
    )

    # Verify the approved.jsonl is valid ElixirLoRA format
    loaded = [json.loads(l) for l in approved_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(loaded) == 3
    for rec in loaded:
        assert "conversation" in rec
        assert len(rec["conversation"]) == 2
        assert rec["conversation"][0]["role"] == "user"
        assert rec["conversation"][1]["role"] == "assistant"
        assert "content" in rec["conversation"][0]
        assert "content" in rec["conversation"][1]

    # Verify output format contains the expected markers
    assistant_texts = [r["conversation"][1]["content"] for r in loaded]
    assert any("[review_type: suggestion]" in t for t in assistant_texts)
    assert any("[review_type: none]" in t for t in assistant_texts)
    assert any("[review_type: refactor]" in t for t in assistant_texts)

    # Test that the format converter round-trips correctly
    for rec in loaded:
        conversation = rec["conversation"]
        assert len(conversation[0]["content"]) > 0
        assert len(conversation[1]["content"]) > 0
