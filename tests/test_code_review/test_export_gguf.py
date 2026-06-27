"""Tests for the standalone export_gguf.py wrapper."""

import subprocess
import sys
from pathlib import Path


def test_export_gguf_script_exists():
    """The standalone export script should exist and be executable."""
    script = Path(__file__).resolve().parents[2] / "examples" / "code_review" / "export_gguf.py"
    assert script.exists()


def test_export_gguf_script_help():
    """The script should respond to --help."""
    script = Path(__file__).resolve().parents[2] / "examples" / "code_review" / "export_gguf.py"
    result = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "quantization" in result.stdout
