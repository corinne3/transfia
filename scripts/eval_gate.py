"""Eval gate (dossier 11.6) : échoue si la qualité de la recherche passe sous un seuil.

Tourne sans modèle (config de test : embeddings et reranker from scratch) sur le corpus d'exemple.
En CI, un changement qui dégrade le Recall@5 sous le seuil bloque le merge.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from transfia.config import load_settings  # noqa: E402
from transfia.container import AppContext  # noqa: E402
from transfia.evaluation.dataset import load_dataset  # noqa: E402
from transfia.evaluation.runner import eval_retrieval  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-recall", type=float, default=0.9)
    args = ap.parse_args()
    ctx = AppContext(load_settings("config/test.yaml", data_dir=tempfile.mkdtemp()))
    ctx.index_samples(log=lambda *a: None)
    items = load_dataset(ctx.settings.resolve("data/samples/golden_sample.jsonl"))
    rep = eval_retrieval(ctx.engine, items, {"hybrid": {"mode": "hybrid", "rerank": True}}, k=5)
    recall = rep["configs"]["hybrid"]["recall@k"]["mean"]
    print(f"Recall@5 hybride = {recall} (seuil {args.min_recall})")
    sys.exit(0 if recall >= args.min_recall else 1)


if __name__ == "__main__":
    main()
