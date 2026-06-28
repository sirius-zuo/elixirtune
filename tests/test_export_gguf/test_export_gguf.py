"""Tests for the export_gguf command."""

import sys

import pytest
from typer.testing import CliRunner


def test_export_gguf_requires_fused_model(tmp_path, monkeypatch):
    """Export GGUF must fail when no fused model exists."""
    monkeypatch.chdir(tmp_path)

    # Clear cached modules
    for mod in list(sys.modules.keys()):
        if mod.startswith("commands"):
            del sys.modules[mod]

    from cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["export-gguf", "test"])
    assert result.exit_code != 0
    assert "No fused model" in result.output


def test_export_gguf_rejects_bad_quantization(tmp_path, monkeypatch):
    """Export GGUF must reject invalid quantization values."""
    (tmp_path / "workspaces" / "test" / "fused").mkdir(parents=True)
    (tmp_path / "workspaces" / "test" / "fused" / "dummy").touch()

    for mod in list(sys.modules.keys()):
        if mod.startswith("commands"):
            del sys.modules[mod]

    monkeypatch.chdir(tmp_path)

    from cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["export-gguf", "test", "--quantization", "INVALID"])
    assert result.exit_code != 0
    assert "Unknown quantization" in result.output
