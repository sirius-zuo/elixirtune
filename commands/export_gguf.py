"""Export fused model to GGUF format using mlx_lm.convert."""

import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import typer

app = typer.Typer(context_settings={"allow_interspersed_args": True})

from commands import _ws


QUANTIZATIONS = ["Q4_K_M", "Q5_K_M", "Q8_0"]


def _check_llama_cpp() -> bool:
    """Check if llama.cpp is available for GGUF conversion."""
    return shutil.which("llama-convert-hf-to-gguf.py") is not None or \
           shutil.which("llama-export-lora") is not None


@app.callback(invoke_without_command=True)
def export_gguf(
    ctx: typer.Context,
    domain: str = typer.Argument(...),
    quantization: str = typer.Option("Q4_K_M", help="Quantization method (Q4_K_M, Q5_K_M, Q8_0)",),
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

    out = output_path or (ws / "fused" / f"{domain}.gguf")

    typer.echo(f"Exporting {domain} to GGUF ({quantization})...")
    typer.echo(f"  Input:  {fused}")
    typer.echo(f"  Output: {out}")

    # Check for llama.cpp
    if not _check_llama_cpp():
        typer.echo("")
        typer.echo("llama.cpp is required for GGUF export.", err=True)
        typer.echo("Install it with: git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make", err=True)
        typer.echo("Then ensure llama-convert-hf-to-gguf.py or llama-export-lora is on PATH.", err=True)
        raise typer.Exit(1)

    # Try mlx_lm.convert first (preferred if available)
    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "mlx_lm.convert",
                "--model-dir", str(fused),
                "--outfile", str(out),
                "--quantize", quantization,
            ],
            check=False,
            capture_output=False,
        )
        if result.returncode == 0:
            typer.echo(f"✅ GGUF exported to: {out}")
            return
    except (FileNotFoundError, ModuleNotFoundError):
        pass  # mlx_lm.convert not available, fall through

    # Fallback: use llama.cpp's conversion
    typer.echo("mlx_lm.convert not available, trying llama.cpp...")
    convert_script = None
    for name in ["llama-convert-hf-to-gguf.py", "llama-export-lora"]:
        script = shutil.which(name)
        if script:
            convert_script = script
            break

    if not convert_script:
        typer.echo("Neither mlx_lm.convert nor llama.cpp conversion scripts found.", err=True)
        typer.echo("Please install one of:", err=True)
        typer.echo("  - mlx-lm: pip install mlx-lm>=0.18 (provides mlx_lm.convert)")
        typer.echo("  - llama.cpp: git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make")
        raise typer.Exit(1)

    # Use llama.cpp to convert
    typer.echo(f"Using {convert_script}...")
    result = subprocess.run(
        [sys.executable, convert_script, str(fused), "--outfile", str(out), "--outtype", "f16"],
        check=False,
        capture_output=False,
    )

    if result.returncode != 0:
        typer.echo(f"GGUF export failed.", err=True)
        raise typer.Exit(1)

    # Quantize if needed (llama.cpp converts to f16 by default)
    if quantization != "Q8_0":
        quantize = shutil.which("llama-quantize") or shutil.which("quantize")
        if quantize:
            typer.echo(f"Quantizing to {quantization}...")
            result = subprocess.run(
                [quantize, str(out), str(out), quantization],
                check=False,
                capture_output=False,
            )
            if result.returncode != 0:
                typer.echo(f"Quantization failed.", err=True)
                raise typer.Exit(1)
        else:
            typer.echo("llama-quantize not found. Output is f16. Install llama.cpp for quantization.",
                       err=True)

    typer.echo(f"✅ GGUF exported to: {out}")
