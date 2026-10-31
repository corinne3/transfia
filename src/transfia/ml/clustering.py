"""Apprentissage NON SUPERVISÉ (F3) : découvrir les grands sujets du corpus, sans étiquettes.

1. RÉDUCTION DE DIMENSION : les embeddings (ex. 384 dimensions) -> 2D pour les visualiser.
   - PCA (linéaire, maximise la variance conservée) : toujours disponible ;
   - UMAP (non linéaire, préserve les voisinages) : optionnel, souvent plus lisible.
2. CLUSTERING :
   - KMeans : il faut choisir k ; on teste plusieurs k et on garde le meilleur SCORE SILHOUETTE
     (proche de 1 = groupes compacts et bien séparés) ;
   - HDBSCAN : trouve tout seul le nombre de groupes, et laisse les points isolés en "bruit" (-1).
3. NOMMAGE des clusters : c-TF-IDF codé from scratch (les mots sur-représentés dans un cluster
   par rapport aux autres), ou un LLM qui lit les passages représentatifs.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from transfia.core.models import Chunk
from transfia.core.text import tokenize


def project_2d(X: np.ndarray, method: str = "pca", seed: int = 42) -> np.ndarray:
    if method == "umap":
        try:
            import umap

            return umap.UMAP(n_components=2, random_state=seed, metric="cosine").fit_transform(X)
        except ImportError:
            pass
    return PCA(n_components=2, random_state=seed).fit_transform(X)


def best_kmeans(X: np.ndarray, k_range=range(2, 9), seed: int = 42) -> tuple[np.ndarray, int, dict]:
    scores = {}
    best = (None, 0, -1.0)
    for k in k_range:
        if k >= len(X):
            break
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        s = float(silhouette_score(X, labels, metric="cosine"))
        scores[k] = round(s, 3)
        if s > best[2]:
            best = (labels, k, s)
    return best[0], best[1], scores


def hdbscan_labels(X: np.ndarray, min_cluster_size: int = 4) -> np.ndarray:
    return HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean").fit_predict(X)


# mots sans valeur pour NOMMER un cluster ("et al.", anglais des références, restes d'URL et de tableaux)
NAME_NOISE = {"al", "and", "of", "the", "in", "for", "to", "with", "on", "by", "from", "https", "http", "doi",
              "org", "www", "br", "pp", "vol", "fig", "tab", "col"}


def c_tf_idf(texts: list[str], labels: np.ndarray, n: int = 6) -> dict[int, list[str]]:
    """c-TF-IDF : on concatène les textes de chaque cluster en UN document, puis
    poids(t, c) = tf(t, c) × log(1 + A / f(t)), avec A = nb moyen de mots par cluster et
    f(t) = fréquence totale de t. Les mots propres à un cluster ressortent."""
    per_cluster: dict[int, Counter] = {}
    for t, lab in zip(texts, labels):
        if lab == -1:
            continue
        per_cluster.setdefault(int(lab), Counter()).update(
            w for w in tokenize(t, stem=False) if w not in NAME_NOISE and not w.isdigit())
    total = Counter()
    for c in per_cluster.values():
        total.update(c)
    A = np.mean([sum(c.values()) for c in per_cluster.values()]) if per_cluster else 1
    out = {}
    for lab, counts in per_cluster.items():
        size = sum(counts.values())
        weights = {w: (f / size) * np.log(1 + A / total[w]) for w, f in counts.items()}
        out[lab] = [w for w, _ in sorted(weights.items(), key=lambda x: -x[1])[:n]]
    return out


def theme_map(chunks: list[Chunk], embedder, method: str = "kmeans", projection: str = "pca") -> dict:
    """Carte des thèmes pour le front : points 2D + nom de chaque cluster."""
    texts = [c.text for c in chunks]
    X = embedder.embed_documents(texts)
    if method == "hdbscan":
        labels, k, scores = hdbscan_labels(X), None, {}
    else:
        labels, k, scores = best_kmeans(X)
    xy = project_2d(X, projection)
    names = c_tf_idf(texts, labels)
    return {
        "method": method, "k": k, "silhouette_by_k": scores,
        "clusters": {str(lab): words for lab, words in names.items()},
        "points": [{"x": float(p[0]), "y": float(p[1]), "cluster": int(lab), "chunk_id": c.id,
                    "title": c.meta.get("title", "")} for p, lab, c in zip(xy, labels, chunks)],
    }
