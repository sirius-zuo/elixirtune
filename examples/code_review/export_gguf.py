#!/usr/bin/env python3
"""Standalone GGUF export for the code-review domain.

Usage:
    python3 examples/code_review/export_gguf.py --domain code-review
    python3 examples/code_review/export_gguf.py --domain code-review --quantization Q5_K_M
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export code-review domain model to GGUF format"
    )
    parser.add_argument("--domain", default="code-review", help="Domain name")
    parser.add_argument(
        "--quantization",
        default="Q4_K_M",
        choices=["Q4_K_M", "Q5_K_M", "Q8_0"],
        help="Quantization method (default: Q4_K_M)",
    )
    parser.add_argument(
        "--root", default=".", help="Project root directory"
    )
    args = parser.parse_args()

    root = Path(args.root)
    cmd = [
        sys.executable, str(root / "cli.py"), "export-gguf", args.domain,
        "--quantization", args.quantization,
    ]

    result = subprocess.run(cmd, cwd=str(root))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
