"""Métriques d'évaluation (F11), toutes codées from scratch.

RECHERCHE (le bon passage est-il remonté ?)
- Recall@k   : part des documents pertinents retrouvés dans le top k.
- Precision@k: part du top k qui est pertinente.
- MRR        : moyenne de 1/rang du premier document pertinent (1 = toujours premier).
- nDCG@k     : gain cumulé "actualisé" (un bon document en rang 1 compte plus qu'en rang 5),
               normalisé par le classement idéal.

GÉNÉRATION
- Faithfulness : part des phrases soutenues par leurs sources (voir rag/verify.py).
- Token F1     : recouvrement de mots avec la réponse de référence (métrique de SQuAD).
- Answer relevance : similarité (embeddings) entre la question et la réponse.

STATISTIQUES (11.3 : un écart de 3 points sur 50 questions ne prouve rien)
- Intervalle de Wilson pour une proportion ; bootstrap pour une moyenne ;
- test de permutation apparié pour comparer deux configurations sur les MÊMES questions.

ACCORD ENTRE ANNOTATEURS (humain vs juge LLM)
- Kappa de Cohen : accord corrigé du hasard (0 = hasard, 1 = parfait) ;
- Corrélation de Spearman : accord sur l'ORDRE des notes.
"""
from __future__ import annotations

import math
import random
from collections import Counter

import numpy as np

from transfia.core.text import tokenize


# ----------------------------------------------------------------- recherche
def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return float("nan")
    return len(set(retrieved[:k]) & relevant) / len(relevant)


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    top = retrieved[:k]
    return len([r for r in top if r in relevant]) / len(top) if top else 0.0


def hit_rate_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    return float(bool(set(retrieved[:k]) & relevant)) if relevant else float("nan")


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for i, r in enumerate(retrieved, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: set[str] | dict[str, float], k: int) -> float:
    gains = relevant if isinstance(relevant, dict) else {r: 1.0 for r in relevant}
    if not gains:
        return float("nan")
    dcg = sum(gains.get(r, 0.0) / math.log2(i + 1) for i, r in enumerate(retrieved[:k], start=1))
    ideal = sorted(gains.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def dedupe(seq: list[str]) -> list[str]:
    """Plusieurs chunks d'un même article -> une seule entrée (évaluation au niveau article)."""
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ---------------------------------------------------------------- génération
def token_f1(prediction: str, reference: str) -> float:
    p, r = tokenize(prediction), tokenize(reference)
    if not p or not r:
        return float(p == r)
    common = Counter(p) & Counter(r)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(r)
    return 2 * precision * recall / (precision + recall)


def answer_relevance(embedder, question: str, answer: str) -> float:
    return float(embedder.embed_query(question) @ embedder.embed_query(answer))


# --------------------------------------------------------------- statistiques
def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """IC à 95 % d'une proportion ; meilleur que p ± 1,96·√(p(1-p)/n) pour les petits n."""
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def bootstrap_ci(values: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> tuple[float, float]:
    """Rééchantillonnage avec remise : la distribution des moyennes donne l'intervalle."""
    vals = np.array([v for v in values if not math.isnan(v)])
    if len(vals) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(vals, size=(n_boot, len(vals)), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def paired_permutation_test(a: list[float], b: list[float], n_perm: int = 5000, seed: int = 0) -> float:
    """p-value : probabilité d'observer un écart moyen au moins aussi grand si A et B étaient équivalents.

    Sous H0, échanger A et B pour une question ne change rien : on échange au hasard et on compte.
    """
    diffs = np.array(a) - np.array(b)
    observed = abs(diffs.mean())
    rng = random.Random(seed)
    count = 0
    for _ in range(n_perm):
        signs = np.array([1 if rng.random() < 0.5 else -1 for _ in diffs])
        if abs((diffs * signs).mean()) >= observed - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def cohen_kappa(a: list, b: list) -> float:
    """κ = (accord observé − accord attendu par hasard) / (1 − accord attendu)."""
    assert len(a) == len(b) and a
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def _ranks(x: list[float]) -> list[float]:
    order = sorted(range(len(x)), key=lambda i: x[i])
    ranks = [0.0] * len(x)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and x[order[j + 1]] == x[order[i]]:
            j += 1
        for k in range(i, j + 1):                       # ex-æquo : rang moyen
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    ra, rb = np.array(_ranks(a)), np.array(_ranks(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def summarize(values: list[float]) -> dict:
    vals = [v for v in values if not (isinstance(v, float) and math.isnan(v))]
    if not vals:
        return {"mean": None, "n": 0}
    lo, hi = bootstrap_ci(vals)
    return {"mean": round(float(np.mean(vals)), 4), "ci95": [round(lo, 4), round(hi, 4)], "n": len(vals)}
