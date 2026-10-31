"""Moteur de recherche complet (F4) : dense + BM25 -> RRF -> reranking.

Modes, pour COMPARER dans le banc d'éval :
- "dense"  : embeddings seuls
- "bm25"   : lexical seul
- "hybrid" : les deux, fusionnés par RRF (+ reranking si configuré)
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from transfia.core.models import Hit
from transfia.observability.tracing import span
from transfia.retrieval.bm25 import BM25
from transfia.retrieval.hybrid import rrf
from transfia.retrieval.query import clean_query
from transfia.retrieval.rerank import NoReranker
from transfia.retrieval.vector_store import _match

Mode = Literal["dense", "bm25", "hybrid"]


class SearchEngine:
    def __init__(self, store, bm25: BM25, embedder, reranker=None, k_dense: int = 20, k_bm25: int = 20,
                 rrf_k: int = 60, rerank_top: int = 20):
        self.store = store
        self.bm25 = bm25
        self.embedder = embedder
        self.reranker = reranker or NoReranker()
        self.k_dense, self.k_bm25, self.rrf_k, self.rerank_top = k_dense, k_bm25, rrf_k, rerank_top

    # -------------------------------------------------------------- briques
    def dense(self, query: str, k: int, filters: dict | None = None, vector: np.ndarray | None = None) -> list[Hit]:
        with span("search.dense", k=k) as s:
            v = vector if vector is not None else self.embedder.embed_query(query)
            hits = self.store.search(v, k, filters)
            s.set(n_hits=len(hits))
            return hits

    def lexical(self, query: str, k: int, filters: dict | None = None) -> list[Hit]:
        with span("search.bm25", k=k) as s:
            results = self.bm25.search(query, k * 3 if filters else k)
            hits = []
            for cid, score in results:
                c = self.store.get(cid)
                if c is not None and _match(c.meta, filters):
                    hits.append(Hit(chunk=c, score=score, source="bm25", rank=len(hits)))
                if len(hits) >= k:
                    break
            s.set(n_hits=len(hits))
            return hits

    # ------------------------------------------------------------ recherche
    def search(self, query: str, k: int = 5, filters: dict | None = None, mode: Mode = "hybrid",
               rerank: bool = True, extra_queries: list[str] | None = None,
               dense_vectors: list[np.ndarray] | None = None) -> list[Hit]:
        """``extra_queries`` : reformulations (multi-query) ; ``dense_vectors`` : vecteurs HyDE."""
        original = query
        query = clean_query(query)                    # "Que dit le corpus sur X ?" -> "X" (bruit retiré)
        with span("search", query=query, original=original, mode=mode, k=k, filters=filters) as s:
            queries = [query] + [clean_query(q) for q in extra_queries or []]
            rankings: list[list[Hit]] = []
            for q in queries:
                if mode in ("dense", "hybrid"):
                    rankings.append(self.dense(q, self.k_dense, filters))
                if mode in ("bm25", "hybrid"):
                    rankings.append(self.lexical(q, self.k_bm25, filters))
            for v in dense_vectors or []:
                rankings.append(self.dense(query, self.k_dense, filters, vector=v))
            fused = rankings[0] if len(rankings) == 1 else rrf(rankings, self.rrf_k)
            if rerank:
                with span("search.rerank", reranker=self.reranker.name, n_in=len(fused)):
                    hits = self.reranker.rerank(query, fused[: self.rerank_top], k)
            else:
                hits = fused[:k]
            s.set(n_hits=len(hits))
            return hits
