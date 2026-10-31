"""Front de Pareto qualité / coût (F14).

Une configuration est DOMINÉE si une autre est au moins aussi bonne sur tous les critères et
strictement meilleure sur au moins un. Le FRONT DE PARETO = les configurations non dominées :
ce sont les seuls choix raisonnables ; le choix entre elles dépend de la priorité métier.
Règle frugale : prendre la configuration la MOINS coûteuse du front qui atteint la qualité cible.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ConfigPoint:
    name: str
    quality: float          # à maximiser (ex. fidélité, F1, note du juge)
    cost: float             # à minimiser (ex. latence, Wh par question, tokens)
    extra: dict | None = None


def dominates(a: ConfigPoint, b: ConfigPoint) -> bool:
    return a.quality >= b.quality and a.cost <= b.cost and (a.quality > b.quality or a.cost < b.cost)


def pareto_front(points: list[ConfigPoint]) -> list[ConfigPoint]:
    front = [p for p in points if not any(dominates(q, p) for q in points if q is not p)]
    return sorted(front, key=lambda p: p.cost)


def cheapest_meeting(points: list[ConfigPoint], min_quality: float) -> ConfigPoint | None:
    ok = [p for p in pareto_front(points) if p.quality >= min_quality]
    return min(ok, key=lambda p: p.cost) if ok else None


def plot(points: list[ConfigPoint], path: str, xlabel: str = "coût", ylabel: str = "qualité") -> str | None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    front = pareto_front(points)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter([p.cost for p in points], [p.quality for p in points], color="#9aa5b1", label="configurations")
    ax.plot([p.cost for p in front], [p.quality for p in front], "-o", color="#2563eb", label="front de Pareto")
    for p in points:
        ax.annotate(p.name, (p.cost, p.quality), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
