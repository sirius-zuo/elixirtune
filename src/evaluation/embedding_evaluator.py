from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-10)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-10)
    return a @ b.T


def compute_similarity(
    anchor: str,
    candidates: list[str],
    model,
    tokenizer,
) -> list[float]:
    """Return cosine similarity scores between anchor and each candidate."""
    from mlx_tune.embeddings import FastEmbeddingModel
    model = FastEmbeddingModel.for_inference(model)
    anchor_emb = model.encode([anchor])
    cand_embs = model.encode(candidates)
    sims = _cosine_similarity(anchor_emb, cand_embs)[0]
    return [float(s) for s in sims]


def recall_at_k(
    val_path: Path,
    model,
    tokenizer,
    k: list[int] = [1, 5, 10],
) -> dict[str, float]:
    """
    Compute Recall@K over a validation set of anchor/positive pairs.

    For each anchor, rank all positives by cosine similarity. A hit at K means
    the correct positive is in the top-K ranked candidates.
    """
    from mlx_tune.embeddings import FastEmbeddingModel
    records = json.loads(Path(val_path).read_text())
    if not records:
        return {f"recall@{ki}": 0.0 for ki in k}

    anchors = [r["anchor"] for r in records]
    positives = [r["positive"] for r in records]

    model = FastEmbeddingModel.for_inference(model)
    anchor_embs = model.encode(anchors)
    positive_embs = model.encode(positives)

    sim_matrix = _cosine_similarity(anchor_embs, positive_embs)

    results = {}
    for ki in k:
        hits = 0
        for i in range(len(anchors)):
            top_k_indices = np.argsort(sim_matrix[i])[::-1][:ki]
            if i in top_k_indices:
                hits += 1
        results[f"recall@{ki}"] = hits / len(anchors)
    return results


def run_beir(dataset_name: str, model, tokenizer) -> dict[str, float]:
    """
    Run BEIR benchmark evaluation for the given dataset.

    Returns {"ndcg@10": float, "recall@100": float}.
    If the `beir` package is not installed, returns {"error": "beir package not installed"}.
    """
    try:
        from beir import util as beir_util
        from beir.datasets.data_loader import GenericDataLoader
        from beir.retrieval.evaluation import EvaluateRetrieval
        from beir.retrieval import models as beir_models
    except ImportError:
        return {"error": "beir package not installed"}

    from mlx_tune.embeddings import FastEmbeddingModel

    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
    data_path = beir_util.download_and_unzip(url, "beir_datasets")
    corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split="test")

    model = FastEmbeddingModel.for_inference(model)

    class _MLXRetriever:
        def encode_corpus(self, corpus_list, **kwargs):
            texts = [
                (d.get("title", "") + " " + d.get("text", "")).strip()
                for d in corpus_list
            ]
            return model.encode(texts)

        def encode_queries(self, queries_list, **kwargs):
            return model.encode(queries_list)

    retriever = EvaluateRetrieval(_MLXRetriever(), score_function="cos_sim")
    results = retriever.retrieve(corpus, queries)
    ndcg, _map, recall, _precision = EvaluateRetrieval.evaluate(
        qrels, results, [10, 100]
    )
    return {
        "ndcg@10": ndcg.get("NDCG@10", 0.0),
        "recall@100": recall.get("Recall@100", 0.0),
    }


def rerank_with_cross_encoder(
    query: str,
    candidates: list[str],
    ce_model_path: str,
) -> list[float]:
    """Rerank candidates using a cross-encoder. Returns scores in original order."""
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(ce_model_path)
    pairs = [(query, c) for c in candidates]
    scores = model.predict(pairs)
    return [float(s) for s in scores]
