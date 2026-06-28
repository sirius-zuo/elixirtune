"""Export fused MLX model to GGUF format."""

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer

app = typer.Typer(context_settings={"allow_interspersed_args": True})

from commands import _ws

QUANTIZATIONS = ["Q4_K_M", "Q5_K_M", "Q8_0", "f16"]

_BUNDLED_CONVERTER = Path(__file__).resolve().parents[1] / "src" / "utils" / "convert_qwen2_to_gguf.py"


def _find_python_with_gguf() -> Path | None:
    """Return the first Python interpreter that has gguf + safetensors + numpy."""
    candidates = [
        Path("/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11"),
        Path("/usr/local/bin/python3.11"),
        shutil.which("python3.11") and Path(shutil.which("python3.11")),
        Path(sys.executable),
        shutil.which("python3") and Path(shutil.which("python3")),
    ]
    for py in candidates:
        if not py or not py.exists():
            continue
        result = subprocess.run(
            [str(py), "-c", "import gguf, safetensors, numpy"],
            capture_output=True,
        )
        if result.returncode == 0:
            return py
    return None


@app.callback(invoke_without_command=True)
def export_gguf(
    ctx: typer.Context,
    domain: str = typer.Argument(...),
    quantization: str = typer.Option("Q4_K_M", help="Quantization: Q4_K_M, Q5_K_M, Q8_0, f16"),
    output_path: Path = typer.Option(None, help="Output GGUF file path"),
) -> None:
    """Export fused model to GGUF format for use with llama.cpp / Ollama."""
    if ctx.invoked_subcommand is not None:
        return

    ws = _ws(domain)
    fused = ws / "fused"

    if not fused.exists() or not any(fused.iterdir()):
        typer.echo(f"No fused model at {fused}. Run: fuse {domain} first.", err=True)
        raise typer.Exit(1)

    if quantization not in QUANTIZATIONS:
        typer.echo(f"Unknown quantization '{quantization}'. Choose from: {QUANTIZATIONS}", err=True)
        raise typer.Exit(1)

    out_f16 = ws / "fused" / f"{domain}_f16.gguf"
    out = output_path or (ws / "fused" / f"{domain}.gguf")

    typer.echo(f"Exporting {domain} to GGUF ({quantization})...")
    typer.echo(f"  Input:  {fused}")
    typer.echo(f"  Output: {out}")

    # Find Python with gguf package (needed for F16 conversion)
    py = _find_python_with_gguf()
    if py is None:
        typer.echo(
            "Could not find Python with 'gguf', 'safetensors', and 'numpy'.\n"
            "Install them: pip install gguf safetensors numpy",
            err=True,
        )
        raise typer.Exit(1)
    typer.echo(f"Using Python: {py}")

    # Step 1: Convert MLX quantized → F16 GGUF using bundled converter
    typer.echo("Step 1: Converting MLX 4-bit model to F16 GGUF...")
    result = subprocess.run(
        [str(py), str(_BUNDLED_CONVERTER), str(fused), str(out_f16)],
        check=False,
    )
    if result.returncode != 0:
        typer.echo("F16 GGUF conversion failed.", err=True)
        raise typer.Exit(1)

    # Step 2: Quantize to target format (unless f16 was requested)
    if quantization == "f16":
        out_f16.rename(out)
        typer.echo(f"✅ GGUF exported to: {out}")
        return

    quantize_bin = shutil.which("llama-quantize")
    if not quantize_bin:
        typer.echo(
            f"llama-quantize not on PATH. F16 GGUF saved at: {out_f16}\n"
            "Install: brew install llama.cpp",
            err=True,
        )
        raise typer.Exit(1)

    typer.echo(f"Step 2: Quantizing to {quantization}...")
    result = subprocess.run(
        [quantize_bin, str(out_f16), str(out), quantization],
        check=False,
    )
    if result.returncode != 0:
        typer.echo("Quantization failed.", err=True)
        raise typer.Exit(1)

    out_f16.unlink(missing_ok=True)
    typer.echo(f"✅ GGUF exported to: {out}")
