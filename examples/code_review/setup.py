#!/usr/bin/env python3
"""Download and convert the github-codereview dataset for ElixirLoRA."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from datasets import load_dataset
from tqdm import tqdm


def sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def make_record(user_content: str, assistant_content: str, meta: dict[str, Any]) -> dict:
    return {
        "conversation": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "meta": meta,
    }


def build_user_message(record: dict) -> str:
    """Build the user prompt from a code review dataset record."""
    parts = ["You are reviewing the following code:\n"]

    diff = record.get("code_diff", "")
    if diff:
        parts.append(f"```diff\n{diff}\n```\n")

    file_path = record.get("file_path", "")
    if file_path:
        parts.append(f"File: {file_path}\n")

    language = record.get("language", "")
    if language:
        parts.append(f"Language: {language}\n")

    pr_title = record.get("pr_title", "")
    if pr_title:
        parts.append(f"PR: {pr_title}\n")

    parts.append("\nPlease provide a code review.")
    return "\n".join(parts)


def build_assistant_message(record: dict) -> str:
    """Build the assistant response from a code review dataset record."""
    review_type = record.get("review_type", "none")
    comment = record.get("comment", "")
    suggestion = record.get("suggestion", "")

    parts = [f"[review_type: {review_type}]"]

    if comment:
        parts.append("")
        parts.append(comment.strip())

    if suggestion:
        parts.append("")
        parts.append(f"```suggestion\n{suggestion}\n```")

    if not (comment or suggestion):
        parts.append(" No issues found.")

    return "\n".join(parts)


def convert_dataset(
    domain: str,
    languages: list[str] | None = None,
    target: int = 200000,
    include_negative: bool = True,
    root: Path = Path("."),
) -> None:
    """Download the github-codereview dataset and convert to ElixirLoRA seed format."""

    dataset_path = root / "data" / "code-review"
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Load dataset from HuggingFace
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset("ronantakizawa/github-codereview", split="train")
    print(f"Loaded {len(dataset)} records.")

    # Filter by languages if specified
    if languages:
        orig = len(dataset)
        dataset = [r for r in dataset if r.get("language", "") in languages]
        print(f"Filtered to {len(dataset)} records in {languages} (was {orig}).")

    # Filter out negative examples if requested
    if not include_negative:
        orig = len(dataset)
        dataset = [r for r in dataset if r.get("review_type") != "none"]
        print(f"Removed negative examples: {len(dataset)} remain (was {orig}).")

    if not dataset:
        print("No records after filtering. Aborting.", file=sys.stderr)
        sys.exit(1)

    # Convert to ElixirLoRA format
    print(f"Converting {len(dataset)} records to ElixirLoRA seed format...")
    seeds = []
    for record in tqdm(dataset, desc="Converting"):
        user_content = build_user_message(record)
        assistant_content = build_assistant_message(record)
        meta = {
            "review_type": record.get("review_type", "none"),
            "language": record.get("language", ""),
            "repo": record.get("repo_name", ""),
            "pr_title": record.get("pr_title", ""),
            "pr_number": record.get("pr_number", 0),
        }
        seeds.append(make_record(user_content, assistant_content, meta))

    # Create workspace
    ws = root / "workspaces" / domain
    seeds_dir = ws / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)

    # Write candidates and approved (same content — seeds from dataset are curated)
    candidates_path = seeds_dir / "candidates.jsonl"
    approved_path = seeds_dir / "approved.jsonl"
    candidates_path.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in seeds) + "\n",
        encoding="utf-8",
    )
    # approved = candidates for this domain (pre-curated dataset)
    approved_path.write_text(
        "\n".join(json.dumps(s, ensure_ascii=False) for s in seeds) + "\n",
        encoding="utf-8",
    )

    # Write config.yaml for code-review domain
    config = {
        "generate": {
            "target_size": min(target, len(seeds)),
            "fewshot_k": 4,
        },
        "filter": {
            "dedup": {
                "embedding_model": "all-MiniLM-L6-v2",
                "similarity_threshold": 0.92,
            },
            "diversity": {
                "quotas": {},  # no specific quotas for code-review
            },
        },
    }
    config_path = ws / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, default_flow_style=False, sort_keys=False),
        encoding="utf-8",
    )

    print(f"\nDone! Created workspace: {ws}")
    print(f"  Seeds: {len(seeds)} records → {approved_path}")
    print(f"  Config: {config_path}")
    print(f"\nNext steps:")
    print(f"  1. (Optional) Edit {candidates_path} to curate seeds")
    print(f"  2. Launch TUI: python3 cli.py tui --domain {domain}")
    print(f"  3. Or run pipeline: python3 cli.py generate {domain}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download code-review dataset for ElixirLoRA")
    parser.add_argument("--domain", default="code-review", help="Workspace domain name")
    parser.add_argument("--languages", default=None, help="Comma-separated list of languages to filter (e.g., 'Python,TypeScript')")
    parser.add_argument("--target", type=int, default=200000, help="Target number of samples")
    parser.add_argument("--no-negative", action="store_true", help="Exclude negative (no issues) examples")
    parser.add_argument("--root", default=".", help="Project root directory")
    args = parser.parse_args()

    languages = [l.strip() for l in args.languages.split(",")] if args.languages else None
    root = Path(args.root)

    convert_dataset(
        domain=args.domain,
        languages=languages,
        target=args.target,
        include_negative=not args.no_negative,
        root=root,
    )


if __name__ == "__main__":
    main()
