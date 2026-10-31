"""Reranking (F4) : le 2e étage de la recherche.

- Étage 1 (retrieval) : BI-ENCODEUR. Requête et documents encodés SÉPARÉMENT -> on peut
  pré-calculer les vecteurs des documents. Rapide, mais approximatif. Objectif : le RAPPEL.
- Étage 2 (reranking) : CROSS-ENCODEUR. Il lit (requête, document) ENSEMBLE et l'attention
  croise chaque mot de la requête avec chaque mot du document. Précis, mais il faut un passage
  du modèle PAR paire : on ne l'applique qu'aux ~20 meilleurs candidats. Objectif : la PRÉCISION.
"""
from __future__ import annotations

from transfia.core.models import Hit
from transfia.core.text import tokenize
from transfia.observability.tracing import span


class NoReranker:
    name = "none"

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        return hits[:top_k]


class LexicalReranker:
    """Baseline from scratch : part des termes de la requête présents dans le passage,
    avec un bonus pour les termes en séquence (bigrammes). Combinée au score d'origine."""

    name = "lexical"

    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha

    def _score(self, q: list[str], d: list[str]) -> float:
        if not q:
            return 0.0
        dset = set(d)
        coverage = sum(t in dset for t in set(q)) / len(set(q))
        qb = set(zip(q, q[1:]))
        db = set(zip(d, d[1:]))
        bigram = len(qb & db) / len(qb) if qb else 0.0
        return 0.8 * coverage + 0.2 * bigram

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        q = tokenize(query)
        max_s = max(h.score for h in hits) or 1.0
        scored = [(self.alpha * self._score(q, tokenize(h.chunk.text)) + (1 - self.alpha) * h.score / max_s, h)
                  for h in hits]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [Hit(chunk=h.chunk, score=s, source="rerank", rank=r) for r, (s, h) in enumerate(scored[:top_k])]


class CrossEncoderReranker:
    def __init__(self, model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1", path: str | None = None,
                 max_length: int = 256):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError('pip install -e ".[local]"') from exc
        # max_length : le coût de l'attention croît avec le CARRÉ de la longueur -> 256 est ~3 à 4x moins cher que 512
        self._model = CrossEncoder(path or model, device="cpu", max_length=max_length)
        self.name = model

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        with span("rerank.cross_encoder", n=len(hits), model=self.name):
            scores = self._model.predict([(query, h.chunk.text) for h in hits])
        order = sorted(zip(scores, hits), key=lambda x: float(x[0]), reverse=True)[:top_k]
        return [Hit(chunk=h.chunk, score=float(s), source="rerank", rank=r) for r, (s, h) in enumerate(order)]


def build_reranker(settings):
    kind = settings.retrieval.reranker
    if kind == "cross-encoder":
        m = settings.retrieval.reranker_model
        return CrossEncoderReranker(m, path=settings.model_path(m), max_length=settings.retrieval.rerank_max_length)
    if kind == "lexical":
        return LexicalReranker()
    return NoReranker()
