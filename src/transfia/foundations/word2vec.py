"""Word2Vec skip-gram avec échantillonnage négatif (NumPy) : comment des EMBEDDINGS s'apprennent.

Hypothèse distributionnelle : « un mot se définit par les mots qui l'entourent ».
- Pour chaque (mot central, mot de contexte) observé dans une fenêtre : rapprocher leurs vecteurs.
- Pour quelques mots tirés au hasard (NÉGATIFS) : les éloigner.
Perte pour une paire positive (c, o) et des négatifs n_i :
    L = −log σ(v_c · u_o) − Σ log σ(−v_c · u_ni)
Les modèles d'embeddings modernes (e5, BGE) suivent la même logique CONTRASTIVE, mais sur des
PHRASES, avec un transformer comme encodeur et des millions de paires (question, passage).
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -20, 20)))


class Word2Vec:
    def __init__(self, dim: int = 24, window: int = 2, negatives: int = 4, lr: float = 0.05, seed: int = 0):
        self.dim, self.window, self.k, self.lr = dim, window, negatives, lr
        self.rng = np.random.default_rng(seed)

    def fit(self, sentences: list[list[str]], epochs: int = 50) -> "Word2Vec":
        counts = Counter(w for s in sentences for w in s)
        self.vocab = sorted(counts)
        self.idx = {w: i for i, w in enumerate(self.vocab)}
        V = len(self.vocab)
        self.Win = self.rng.normal(0, 0.1, (V, self.dim))     # vecteurs "centre"
        self.Wout = np.zeros((V, self.dim))                   # vecteurs "contexte"
        freq = np.array([counts[w] for w in self.vocab], dtype=float) ** 0.75
        self.neg_dist = freq / freq.sum()                     # puissance 3/4 : astuce du papier original
        for _ in range(epochs):
            for s in sentences:
                ids = [self.idx[w] for w in s]
                for i, c in enumerate(ids):
                    for j in range(max(0, i - self.window), min(len(ids), i + self.window + 1)):
                        if j != i:
                            self._update(c, ids[j])
        return self

    def _update(self, c: int, o: int) -> None:
        negs = self.rng.choice(len(self.vocab), size=self.k, p=self.neg_dist)
        targets = np.concatenate([[o], negs])
        labels = np.array([1.0] + [0.0] * self.k)
        v = self.Win[c]
        u = self.Wout[targets]
        g = sigmoid(u @ v) - labels                           # gradient de la perte logistique
        self.Win[c] -= self.lr * g @ u
        self.Wout[targets] -= self.lr * g[:, None] * v[None, :]

    def vector(self, w: str) -> np.ndarray:
        v = self.Win[self.idx[w]]
        return v / (np.linalg.norm(v) + 1e-12)

    def most_similar(self, w: str, n: int = 3) -> list[tuple[str, float]]:
        q = self.vector(w)
        M = self.Win / (np.linalg.norm(self.Win, axis=1, keepdims=True) + 1e-12)
        sims = M @ q
        order = [i for i in np.argsort(-sims) if self.vocab[i] != w][:n]
        return [(self.vocab[i], round(float(sims[i]), 3)) for i in order]


SENTENCES = [s.split() for s in [
    "la vache produit du lait", "la brebis produit du lait", "la chevre produit du lait",
    "la vache mange du fourrage", "la brebis mange du fourrage", "la chevre mange du foin",
    "le porc mange des cereales", "la truie mange des cereales", "le poulet mange des cereales",
    "le veau boit du colostrum", "l agneau boit du colostrum", "le chevreau boit du colostrum",
] * 3]


def demo() -> None:
    m = Word2Vec().fit(SENTENCES, epochs=60)
    for w in ("vache", "porc", "veau"):
        print(f"Voisins de « {w} » :", m.most_similar(w))


if __name__ == "__main__":
    demo()
