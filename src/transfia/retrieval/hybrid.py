"""Fusion de résultats (F4) : Reciprocal Rank Fusion (RRF) et fusion pondérée des scores.

RRF : score(d) = Σ_listes 1 / (k + rang(d)),  avec k ≈ 60 et rang commençant à 1.
- On ne combine que des RANGS : pas besoin de normaliser des scores hétérogènes
  (un score BM25 de 12,3 et une similarité cosinus de 0,82 ne sont pas comparables).
- k amortit l'écart entre les tout premiers rangs.
"""
from __future__ import annotations

from transfia.core.models import Hit


def rrf(rankings: list[list[Hit]], k: int = 60, top: int | None = None) -> list[Hit]:
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for ranking in rankings:
        for rank, hit in enumerate(ranking, start=1):
            cid = hit.chunk.id
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            best.setdefault(cid, hit)
    order = sorted(scores, key=scores.get, reverse=True)[:top]
    return [Hit(chunk=best[c].chunk, score=scores[c], source="hybrid", rank=r) for r, c in enumerate(order)]


def weighted_fusion(rankings: list[list[Hit]], weights: list[float], top: int | None = None) -> list[Hit]:
    """Alternative à comparer : min-max sur chaque liste puis somme pondérée."""
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for ranking, w in zip(rankings, weights):
        if not ranking:
            continue
        vals = [h.score for h in ranking]
        lo, hi = min(vals), max(vals)
        for h in ranking:
            norm = 1.0 if hi == lo else (h.score - lo) / (hi - lo)
            scores[h.chunk.id] = scores.get(h.chunk.id, 0.0) + w * norm
            best.setdefault(h.chunk.id, h)
    order = sorted(scores, key=scores.get, reverse=True)[:top]
    return [Hit(chunk=best[c].chunk, score=scores[c], source="weighted", rank=r) for r, c in enumerate(order)]
