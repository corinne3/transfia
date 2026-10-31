"""Apprentissage par renforcement (RL) et alignement des LLM.

RL : un AGENT agit dans un ENVIRONNEMENT, reçoit des RÉCOMPENSES et apprend une POLITIQUE
qui maximise la récompense cumulée.
1. BANDIT MANCHOT à K bras : le dilemme EXPLORATION / EXPLOITATION.
   ε-greedy : avec une probabilité ε on essaie un bras au hasard, sinon on joue le meilleur connu.
   (Exemple d'application : choisir automatiquement le prompt qui obtient les meilleurs retours.)
2. Q-LEARNING sur une grille : on apprend Q(état, action) = valeur attendue.
       Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]
   γ (gamma) = importance du futur ; α = taux d'apprentissage.

LIEN AVEC LES LLM — l'ALIGNEMENT (après pré-entraînement et SFT) :
- RLHF : des humains comparent des réponses ; on entraîne un MODÈLE DE RÉCOMPENSE, puis on
  optimise le LLM (PPO) pour maximiser cette récompense, sans trop s'éloigner du modèle de départ.
- DPO : on saute le modèle de récompense ; on optimise directement sur les paires
  (réponse préférée, réponse rejetée) avec la perte ci-dessous.
"""
from __future__ import annotations

import numpy as np


def epsilon_greedy_bandit(true_means: list[float], steps: int = 2000, eps: float = 0.1, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    k = len(true_means)
    estimates, counts = np.zeros(k), np.zeros(k)
    total = 0.0
    for _ in range(steps):
        arm = rng.integers(k) if rng.random() < eps else int(np.argmax(estimates))
        reward = rng.normal(true_means[arm], 1.0)
        counts[arm] += 1
        estimates[arm] += (reward - estimates[arm]) / counts[arm]       # moyenne incrémentale
        total += reward
    return {"estimates": estimates.round(2).tolist(), "pulls": counts.astype(int).tolist(),
            "avg_reward": round(total / steps, 3), "best_possible": max(true_means)}


GRID = ["S..#",
        ".#..",
        "...G"]
ACTIONS = [(0, 1), (1, 0), (0, -1), (-1, 0)]


def q_learning(grid=GRID, episodes: int = 500, alpha: float = 0.5, gamma: float = 0.9, eps: float = 0.2,
               seed: int = 0) -> tuple[np.ndarray, list[tuple[int, int]]]:
    rng = np.random.default_rng(seed)
    H, W = len(grid), len(grid[0])
    Q = np.zeros((H, W, len(ACTIONS)))
    start = next((y, x) for y in range(H) for x in range(W) if grid[y][x] == "S")

    def step(s, a):
        ny, nx = s[0] + ACTIONS[a][0], s[1] + ACTIONS[a][1]
        if not (0 <= ny < H and 0 <= nx < W) or grid[ny][nx] == "#":
            return s, -1.0, False                        # mur : on reste sur place, pénalité
        if grid[ny][nx] == "G":
            return (ny, nx), 10.0, True
        return (ny, nx), -0.1, False                     # petit coût par pas : incite au chemin court

    for _ in range(episodes):
        s = start
        for _ in range(50):
            a = rng.integers(len(ACTIONS)) if rng.random() < eps else int(np.argmax(Q[s]))
            s2, r, done = step(s, a)
            target = r if done else r + gamma * Q[s2].max()
            Q[s][a] += alpha * (target - Q[s][a])
            s = s2
            if done:
                break
    # politique apprise (gloutonne)
    path, s = [start], start
    for _ in range(20):
        s, _, done = step(s, int(np.argmax(Q[s])))
        path.append(s)
        if done:
            break
    return Q, path


def dpo_loss(logp_chosen: float, logp_rejected: float, ref_logp_chosen: float, ref_logp_rejected: float,
             beta: float = 0.1) -> float:
    """Perte DPO = −log σ( β [ (log π(y+) − log π_ref(y+)) − (log π(y−) − log π_ref(y−)) ] ).

    Elle diminue quand le modèle augmente la probabilité de la réponse PRÉFÉRÉE (y+) par rapport à
    la rejetée (y−), RELATIVEMENT au modèle de référence (ce qui l'empêche de dériver trop loin).
    """
    margin = beta * ((logp_chosen - ref_logp_chosen) - (logp_rejected - ref_logp_rejected))
    return float(np.log1p(np.exp(-margin)))             # = −log σ(margin), écrit de façon stable


def demo() -> None:
    print("Bandit ε-greedy (3 prompts, vraie qualité 0.2 / 0.5 / 0.8) :", epsilon_greedy_bandit([0.2, 0.5, 0.8]))
    _, path = q_learning()
    print("Q-learning, chemin appris :", path)
    print("DPO : perte si le modèle préfère déjà la bonne réponse :", round(dpo_loss(-5, -9, -6, -6), 4))
    print("DPO : perte si le modèle préfère la mauvaise réponse :   ", round(dpo_loss(-9, -5, -6, -6), 4))


if __name__ == "__main__":
    demo()
