import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import typer
import yaml
from data.synthetic.config import load_config
from data.synthetic.teacher import from_config
from data.dpo.pipeline import run_prepare_dpo

app = typer.Typer(context_settings={"allow_interspersed_args": True})

from commands import _ws


def _judge_score(prompt: str, response: str, teacher) -> int:
    """Score a response 1-5 via the teacher (same rubric as the synth judge)."""
    q = (
        f"INPUT:\n{prompt}\n\nRESPONSE:\n{response}\n\n"
        "Rate the response quality from 1 to 5. Return ONLY the integer."
    )
    try:
        digits = "".join(c for c in teacher.chat([{"role": "user", "content": q}]) if c.isdigit())
        return int(digits[:1])
    except (ValueError, IndexError):
        return 0


def _generate_with_model(model_path: str, prompts: list[str], max_tokens: int, log) -> list[str]:
    import mlx_lm
    log(f"Loading model for candidates: {model_path}")
    model, tok = mlx_lm.load(model_path)
    outs = []
    for i, p in enumerate(prompts):
        try:
            text = tok.apply_chat_template(
                [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
            )
        except Exception:
            text = p
        outs.append(mlx_lm.generate(model, tok, text, max_tokens=max_tokens).strip())
        log(f"  {model_path}: {i + 1}/{len(prompts)}")
    return outs


@app.callback(invoke_without_command=True)
def prepare_dpo(
    ctx: typer.Context,
    domain: str = typer.Argument(...),
    model_config: Path = typer.Option(Path("config/model_config.yaml"), help="Model config (for the base model path)"),
    teacher_samples: int = typer.Option(None, help="Teacher candidate answers per prompt"),
    use_sft: bool = typer.Option(None, help="Include the SFT-fused model as a candidate source"),
    use_base: bool = typer.Option(None, help="Include the base model as a candidate source"),
    min_margin: int = typer.Option(None, help="Min judge-score gap to keep a pair"),
    max_prompts: int = typer.Option(None, help="Cap on prompts to use"),
    max_tokens: int = typer.Option(None, help="Max tokens per generated candidate"),
):
    """Build DPO preference data (processed/dpo.json) from configurable candidate sources."""
    if ctx.invoked_subcommand is not None:
        return

    cfg = load_config(domain)
    dcfg = cfg.get("dpo_data", {})
    # CLI flags override config; config overrides built-in defaults.
    teacher_samples = dcfg.get("teacher_samples", 2) if teacher_samples is None else teacher_samples
    use_sft = dcfg.get("use_sft", True) if use_sft is None else use_sft
    use_base = dcfg.get("use_base", True) if use_base is None else use_base
    min_margin = dcfg.get("min_margin", 2) if min_margin is None else min_margin
    max_prompts = dcfg.get("max_prompts", 200) if max_prompts is None else max_prompts
    max_tokens = dcfg.get("max_tokens", 256) if max_tokens is None else max_tokens

    teacher = from_config(cfg)
    base_path = yaml.safe_load(Path(model_config).read_text())["base_model"]["path"]
    fused_dir = _ws(domain) / "fused"

    def log(msg):
        typer.echo(msg)

    def gather(prompts: list[str]) -> list[list[str]]:
        cands: list[list[str]] = [[] for _ in prompts]
        for k in range(teacher_samples):
            log(f"Teacher candidate pass {k + 1}/{teacher_samples}…")
            for i, p in enumerate(prompts):
                cands[i].append(teacher.chat([{"role": "user", "content": p}], temperature=0.9))
        if use_base:
            for i, txt in enumerate(_generate_with_model(base_path, prompts, max_tokens, log)):
                cands[i].append(txt)
        if use_sft:
            if fused_dir.exists() and any(fused_dir.iterdir()):
                for i, txt in enumerate(_generate_with_model(str(fused_dir), prompts, max_tokens, log)):
                    cands[i].append(txt)
            else:
                log("SFT source requested but no fused model found; skipping it. Fuse after SFT.")
        return cands

    run_prepare_dpo(
        domain,
        gather_candidates=gather,
        judge_fn=lambda p, r: _judge_score(p, r, teacher),
        min_margin=min_margin,
        max_prompts=max_prompts,
        log=log,
    )
