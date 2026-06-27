import json
import sys
from pathlib import Path

import pytest
from unittest.mock import patch


def test_make_record_shape():
    """Test that make_record produces the correct structure."""
    from examples.code_review.setup import make_record
    rec = make_record("user content", "assistant content", {"source": "test"})
    assert rec["conversation"] == [
        {"role": "user", "content": "user content"},
        {"role": "assistant", "content": "assistant content"},
    ]
    assert rec["meta"]["source"] == "test"


def test_build_user_message_with_all_fields():
    """Test user message construction with all fields present."""
    from examples.code_review.setup import build_user_message
    record = {
        "code_diff": "@@ -1,3 +1,3 @@\n-old\n+new",
        "file_path": "src/example.py",
        "language": "Python",
        "pr_title": "Fix example",
    }
    result = build_user_message(record)
    assert "You are reviewing the following code:" in result
    assert "```diff" in result
    assert "@@ -1,3 +1,3 @@" in result
    assert "File: src/example.py" in result
    assert "Language: Python" in result
    assert "PR: Fix example" in result
    assert "Please provide a code review." in result


def test_build_user_message_minimal():
    """Test user message with empty fields."""
    from examples.code_review.setup import build_user_message
    record = {"code_diff": "", "file_path": "", "language": "", "pr_title": ""}
    result = build_user_message(record)
    assert "You are reviewing the following code:" in result
    assert "Please provide a code review." in result


def test_build_assistant_message_with_review():
    """Test assistant message for a suggestion-type review."""
    from examples.code_review.setup import build_assistant_message
    record = {
        "review_type": "suggestion",
        "comment": "Consider using a more descriptive name.",
        "suggestion": "def compute_total_price():",
    }
    result = build_assistant_message(record)
    assert "[review_type: suggestion]" in result
    assert "Consider using a more descriptive name." in result
    assert "```suggestion" in result
    assert "def compute_total_price():" in result


def test_build_assistant_message_negative():
    """Test assistant message for no-issues (negative) example."""
    from examples.code_review.setup import build_assistant_message
    record = {
        "review_type": "none",
        "comment": "",
        "suggestion": "",
    }
    result = build_assistant_message(record)
    assert "[review_type: none]" in result
    assert "No issues found." in result


def test_build_assistant_message_with_only_comment():
    """Test assistant message when only comment is present (no suggestion)."""
    from examples.code_review.setup import build_assistant_message
    record = {
        "review_type": "question",
        "comment": "Why is this check necessary?",
        "suggestion": "",
    }
    result = build_assistant_message(record)
    assert "[review_type: question]" in result
    assert "Why is this check necessary?" in result
    assert "```suggestion" not in result


def test_build_assistant_message_with_only_suggestion():
    """Test assistant message when only suggestion is present."""
    from examples.code_review.setup import build_assistant_message
    record = {
        "review_type": "refactor",
        "comment": "",
        "suggestion": "def helper_func(x):\n    return x * 2",
    }
    result = build_assistant_message(record)
    assert "[review_type: refactor]" in result
    assert "```suggestion" in result
    assert "def helper_func(x):" in result


def test_write_read_jsonl_roundtrip(tmp_path):
    """Test that seeds can be written and read back correctly."""
    from examples.code_review.setup import make_record
    rec = make_record("user", "assistant", {"review_type": "suggestion"})
    path = tmp_path / "seeds" / "approved.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(rec, ensure_ascii=False) + "\n", encoding="utf-8")
    lines = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert len(lines) == 1
    assert lines[0] == rec


def test_config_yaml_structure(tmp_path):
    """Test that the generated config.yaml has the expected structure."""
    import examples.code_review.setup as setup_mod

    fake_records = [
        {
            "code_diff": "@@ -1,3 +1,3 @@\n-old\n+new",
            "file_path": "src/example.py",
            "language": "Python",
            "pr_title": "Fix example",
            "review_type": "suggestion",
            "comment": "Fix this.",
            "suggestion": "def fixed():\n    pass",
            "repo_name": "test/repo",
            "pr_number": 1,
        },
        {
            "code_diff": "@@ -1,0 +1,1 @@\n+added",
            "file_path": "src/other.py",
            "language": "JavaScript",
            "pr_title": "Add feature",
            "review_type": "refactor",
            "comment": "Extract this.",
            "suggestion": "function helper():\n    pass",
            "repo_name": "test/repo2",
            "pr_number": 2,
        },
    ]

    with patch.object(setup_mod, "load_dataset", return_value=fake_records):
        setup_mod.convert_dataset(domain="test-domain", target=100, include_negative=False, root=tmp_path)

    import yaml
    config_path = tmp_path / "workspaces" / "test-domain" / "config.yaml"
    config = yaml.safe_load(config_path.read_text())

    assert "generate" in config
    assert "target_size" in config["generate"]
    assert config["generate"]["target_size"] == 2  # min(100, 2 fake records)
    assert "filter" in config
    assert "dedup" in config["filter"]
    assert config["filter"]["dedup"]["embedding_model"] == "all-MiniLM-L6-v2"
