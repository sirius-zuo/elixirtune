"""Tests for the export_gguf command."""

import pytest
from typer.testing import CliRunner


def test_export_gguf_requires_fused_model(tmp_path, monkeypatch):
    """Export GGUF must fail when no fused model exists."""
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_root / "src"))

    # Clear cached modules
    for mod in list(sys.modules.keys()):
        if mod.startswith("commands"):
            del sys.modules[mod]

    monkeypatch.chdir(tmp_path)

    from cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["export-gguf", "test"])
    assert result.exit_code != 0
    assert "No fused model" in result.output


def test_export_gguf_rejects_bad_quantization(tmp_path, monkeypatch):
    """Export GGUF must reject invalid quantization values."""
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(_root / "src"))

    for mod in list(sys.modules.keys()):
        if mod.startswith("commands"):
            del sys.modules[mod]

    monkeypatch.chdir(tmp_path)

    (tmp_path / "workspaces" / "test" / "fused").mkdir(parents=True)
    (tmp_path / "workspaces" / "test" / "fused" / "dummy").touch()

    from cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["export-gguf", "test", "--quantization", "INVALID"])
    assert result.exit_code != 0
    assert "Unknown quantization" in result.output
