"""Routage et cascade de modèles (F14, frugalité).

Idée : le petit modèle répond par défaut ; on ne paie le gros modèle que si nécessaire.
Deux stratégies classiques :
1. ROUTAGE a priori : on estime la difficulté de la question AVANT de répondre (heuristique ici ;
   en production, souvent un petit classifieur entraîné).
2. CASCADE a posteriori : le petit modèle répond, on vérifie (ex. fidélité aux sources),
   et on relance avec le gros modèle si la vérification échoue.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from transfia.core.text import normalize

_HARD_MARKERS = (
    "compare", "comparer", "difference", "pourquoi", "explique", "expliquer", "analyse", "synthese",
    "avantages et inconvenients", "consequences", "relation entre", "impact", "evaluer", "plan", "sequence",
)


@dataclass
class RouteDecision:
    model: str
    difficulty: float
    reason: str


def difficulty_score(question: str) -> float:
    """Score heuristique entre 0 et 1 (longueur, marqueurs de raisonnement, questions multiples)."""
    q = normalize(question)
    words = len(q.split())
    score = min(words / 40, 1.0) * 0.4
    score += 0.4 * min(sum(m in q for m in _HARD_MARKERS), 2) / 2
    score += 0.2 * min(q.count("?") + len(re.findall(r"\bet\b", q)) / 3, 1.0)
    return round(min(score, 1.0), 3)


class ModelRouter:
    def __init__(self, small_model: str, big_model: str | None, threshold: float = 0.5):
        self.small = small_model
        self.big = big_model
        self.threshold = threshold

    def route(self, question: str) -> RouteDecision:
        d = difficulty_score(question)
        if self.big and d >= self.threshold:
            return RouteDecision(self.big, d, f"difficulté {d} >= {self.threshold}")
        return RouteDecision(self.small, d, f"difficulté {d} < {self.threshold}" if self.big else "pas de gros modèle")

    def escalate(self, faithfulness: float | None, min_faithfulness: float = 0.6) -> str | None:
        """Cascade : renvoie le gros modèle si la réponse du petit est jugée peu fidèle."""
        if self.big and faithfulness is not None and faithfulness < min_faithfulness:
            return self.big
        return None
