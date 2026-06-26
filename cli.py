import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import typer
from data.synthetic.config import load_config
from data.synthetic.bootstrap import bootstrap_seeds
from data.synthetic.teacher import from_config
from data.synthetic.embedder import SentenceTransformerEmbedder
from data.synthetic.pipeline import run_generate
from data.synthetic.io import read_jsonl, write_jsonl

app = typer.Typer()


def _ws(domain: str) -> Path:
    return Path("workspaces") / domain


@app.command()
def init(domain: str, desc: str = typer.Option(None), seeds: str = typer.Option(None)):
    cand = _ws(domain) / "seeds" / "candidates.jsonl"
    cand.parent.mkdir(parents=True, exist_ok=True)
    if seeds:
        recs = read_jsonl(seeds)
        write_jsonl(cand, recs)
        typer.echo(f"Imported {len(recs)} seeds to {cand}")
    else:
        if not desc:
            raise typer.BadParameter("provide --seeds PATH or --desc TEXT")
        cfg = load_config(domain)
        teacher = from_config(cfg)
        recs = bootstrap_seeds(desc, teacher, cfg["bootstrap"]["starter_count"])
        write_jsonl(cand, recs)
        typer.echo(f"Bootstrapped {len(recs)} candidate seeds to {cand}. Edit, then run: curate {domain}")


@app.command()
def curate(domain: str):
    cand = _ws(domain) / "seeds" / "candidates.jsonl"
    approved = _ws(domain) / "seeds" / "approved.jsonl"
    shutil.copyfile(cand, approved)
    typer.echo(f"Approved {len(read_jsonl(approved))} seeds → {approved}")


@app.command()
def generate(domain: str):
    cfg = load_config(domain)
    teacher = from_config(cfg)
    embedder = SentenceTransformerEmbedder(cfg["filter"]["dedup"]["embedding_model"])
    run_dir = run_generate(domain, cfg, teacher, embedder)
    typer.echo(f"Done. Run artifacts in {run_dir}")


@app.command()
def prepare(domain: str, system_prompt: str = typer.Option(..., help="System prompt for the fine-tuned model")):
    """Convert filtered JSONL from generate into train/val/test splits for the training pipeline."""
    filtered = _ws(domain) / "generated" / "filtered.jsonl"
    if not filtered.exists():
        typer.echo(f"No filtered data found at {filtered}. Run: generate {domain}", err=True)
        raise typer.Exit(1)

    from data.preprocessor import DataPreprocessor
    import yaml

    data_cfg_path = Path("config/data_config.yaml")
    data_cfg = yaml.safe_load(data_cfg_path.read_text()) if data_cfg_path.exists() else {}
    test_split = data_cfg.get("dataset", {}).get("test_split", 0.1)
    val_split = data_cfg.get("dataset", {}).get("val_split", 0.1)

    preprocessor = DataPreprocessor(system_prompt=system_prompt, test_split_ratio=test_split, val_split_ratio=val_split)

    records = read_jsonl(filtered)
    # DataPreprocessor.extract_conversations expects a HF dataset-like object;
    # replicate its logic directly over our list[dict] to avoid the HF dependency here.
    samples = []
    for rec in records:
        conversation = rec["conversation"]
        for i in range(0, len(conversation) - 1, 2):
            q = conversation[i]["content"]
            a = conversation[i + 1]["content"]
            samples.append(preprocessor.format_conversation_sample(q, a))

    import random
    random.shuffle(samples)
    train, val, test = preprocessor.create_train_val_test_split(samples)

    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    for name, split in [("train", train), ("val", val), ("test", test)]:
        (out / f"{name}.json").write_text(json.dumps(split, indent=2))
    stats = {"train_size": len(train), "val_size": len(val), "test_size": len(test),
             "total_size": len(samples)}
    (out / "data_stats.json").write_text(json.dumps(stats, indent=2))
    typer.echo(f"Prepared {len(samples)} samples → {out}  (train={len(train)}, val={len(val)}, test={len(test)})")


if __name__ == "__main__":
    app()
