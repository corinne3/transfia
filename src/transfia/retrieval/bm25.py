"""BM25 codé from scratch (F4) : le score lexical de référence des moteurs de recherche.

Pour une requête q et un document d :

    score(d, q) = Σ_{t ∈ q}  IDF(t) · f(t,d)·(k1+1) / ( f(t,d) + k1·(1 - b + b·|d|/avgdl) )

- f(t,d)  : nombre d'occurrences du terme t dans d
- IDF(t)  = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 ) : un terme RARE pèse plus
- k1 (~1.2-2) : SATURATION. La 10e occurrence apporte beaucoup moins que la 1re.
- b (~0.75)   : NORMALISATION par la longueur. Un long document n'est pas favorisé
                juste parce qu'il contient plus de mots.

Force : termes exacts (noms de races, molécules, sigles). Faiblesse : ignore les synonymes.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from transfia.core.text import tokenize


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.ids: list[str] = []
        self.doc_tf: list[Counter] = []
        self.doc_len: list[int] = []
        self.df: Counter = Counter()
        self.avgdl = 0.0

    def fit(self, ids: list[str], texts: list[str]) -> "BM25":
        self.ids, self.doc_tf, self.doc_len, self.df = [], [], [], Counter()
        return self.add(ids, texts)

    def add(self, ids: list[str], texts: list[str]) -> "BM25":
        for i, t in zip(ids, texts):
            toks = tokenize(t)
            tf = Counter(toks)
            self.ids.append(i)
            self.doc_tf.append(tf)
            self.doc_len.append(len(toks))
            self.df.update(tf.keys())
        self.avgdl = float(np.mean(self.doc_len)) if self.doc_len else 0.0
        return self

    def idf(self, term: str) -> float:
        n, df = len(self.ids), self.df.get(term, 0)
        return math.log((n - df + 0.5) / (df + 0.5) + 1.0)

    def scores(self, query: str) -> np.ndarray:
        q_terms = tokenize(query)
        out = np.zeros(len(self.ids), dtype=np.float32)
        if not q_terms or not self.ids:
            return out
        for term in set(q_terms):
            if term not in self.df:
                continue
            idf = self.idf(term)
            for j, tf in enumerate(self.doc_tf):
                f = tf.get(term, 0)
                if f:
                    denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[j] / self.avgdl)
                    out[j] += idf * f * (self.k1 + 1) / denom
        return out

    def search(self, query: str, k: int = 10) -> list[tuple[str, float]]:
        s = self.scores(query)
        order = np.argsort(-s)[:k]
        return [(self.ids[i], float(s[i])) for i in order if s[i] > 0]

    # Persistance en JSON (pas de pickle : un pickle peut exécuter du code au chargement, 11.5)
    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"k1": self.k1, "b": self.b, "ids": self.ids,
                                    "tf": [dict(c) for c in self.doc_tf]}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BM25":
        d = json.loads(path.read_text(encoding="utf-8"))
        bm = cls(d["k1"], d["b"])
        for i, tf in zip(d["ids"], d["tf"]):
            c = Counter(tf)
            bm.ids.append(i)
            bm.doc_tf.append(c)
            bm.doc_len.append(sum(c.values()))
            bm.df.update(c.keys())
        bm.avgdl = float(np.mean(bm.doc_len)) if bm.doc_len else 0.0
        return bm
