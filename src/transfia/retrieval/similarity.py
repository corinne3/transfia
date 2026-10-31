"""Similarité cosinus et top-k, codés à la main (F4).

cos(a, b) = a·b / (‖a‖ ‖b‖), entre -1 (opposés) et 1 (même direction).
Si les vecteurs sont déjà normalisés (norme 1), cos(a, b) = a·b : un simple produit scalaire,
et pour N documents d'un coup, un produit matrice-vecteur.
"""
from __future__ import annotations

import math

import numpy as np


def cosine_python(a: list[float], b: list[float]) -> float:
    """Version "à la main", sans NumPy, pour comprendre la formule."""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


def cosine_matrix(queries: np.ndarray, docs: np.ndarray) -> np.ndarray:
    """Similarités de chaque requête avec chaque document (vectorisé)."""
    q = queries / np.maximum(np.linalg.norm(queries, axis=-1, keepdims=True), 1e-12)
    d = docs / np.maximum(np.linalg.norm(docs, axis=-1, keepdims=True), 1e-12)
    return q @ d.T


def top_k(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices des k meilleurs scores, du meilleur au moins bon.

    ``argpartition`` trouve les k meilleurs en O(N) sans tout trier (O(N log N)),
    puis on ne trie que ces k-là.
    """
    k = min(k, len(scores))
    if k <= 0:
        return np.array([], dtype=int)
    idx = np.argpartition(-scores, k - 1)[:k]
    return idx[np.argsort(-scores[idx])]
