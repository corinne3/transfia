"""Réseau de neurones (perceptron multicouche) codé from scratch, avec rétropropagation.

    x --[W1,b1]--> z1 --ReLU--> a1 --[W2,b2]--> z2 --softmax--> p

PROPAGATION AVANT : on calcule la sortie couche par couche.
PERTE : entropie croisée L = −log p[classe correcte].
RÉTROPROPAGATION (règle de dérivation en chaîne), de la sortie vers l'entrée :
    dz2 = p − y_onehot                 (résultat remarquable softmax + entropie croisée)
    dW2 = a1ᵀ dz2 / n,  db2 = Σ dz2 / n
    da1 = dz2 W2ᵀ,  dz1 = da1 ⊙ 1[z1 > 0]   (dérivée de ReLU)
    dW1 = xᵀ dz1 / n,  db1 = Σ dz1 / n
OPTIMISEURS : SGD (w −= lr·g) et ADAM (moyennes mobiles du gradient et de son carré, avec
correction de biais) — Adam est l'optimiseur par défaut pour entraîner les transformers.

``gradient_check`` compare le gradient analytique au gradient NUMÉRIQUE (différences finies) :
LA technique pour vérifier qu'une rétropropagation est juste.
"""
from __future__ import annotations

import numpy as np


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)          # stabilité numérique : exp de grands nombres = overflow
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, n_in: int, n_hidden: int, n_out: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        # initialisation de He (adaptée à ReLU) : variance 2 / n_in
        self.params = {
            "W1": rng.normal(0, np.sqrt(2 / n_in), (n_in, n_hidden)), "b1": np.zeros(n_hidden),
            "W2": rng.normal(0, np.sqrt(2 / n_hidden), (n_hidden, n_out)), "b2": np.zeros(n_out),
        }
        self._adam = {k: (np.zeros_like(v), np.zeros_like(v)) for k, v in self.params.items()}
        self._t = 0

    def forward(self, X):
        p = self.params
        z1 = X @ p["W1"] + p["b1"]
        a1 = np.maximum(z1, 0)
        z2 = a1 @ p["W2"] + p["b2"]
        return softmax(z2), (X, z1, a1)

    def loss(self, X, y, l2: float = 0.0) -> float:
        probs, _ = self.forward(X)
        ce = -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12))
        return float(ce + l2 * (np.sum(self.params["W1"] ** 2) + np.sum(self.params["W2"] ** 2)))

    def gradients(self, X, y, l2: float = 0.0) -> dict:
        probs, (X, z1, a1) = self.forward(X)
        n = len(y)
        dz2 = probs.copy()
        dz2[np.arange(n), y] -= 1
        dz2 /= n
        g = {"W2": a1.T @ dz2 + 2 * l2 * self.params["W2"], "b2": dz2.sum(0)}
        da1 = dz2 @ self.params["W2"].T
        dz1 = da1 * (z1 > 0)
        g["W1"] = X.T @ dz1 + 2 * l2 * self.params["W1"]
        g["b1"] = dz1.sum(0)
        return g

    def step(self, grads: dict, lr: float = 0.01, optimizer: str = "adam", b1=0.9, b2=0.999, eps=1e-8) -> None:
        self._t += 1
        for k, g in grads.items():
            if optimizer == "sgd":
                self.params[k] -= lr * g
            else:
                m, v = self._adam[k]
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g ** 2
                self._adam[k] = (m, v)
                m_hat, v_hat = m / (1 - b1 ** self._t), v / (1 - b2 ** self._t)
                self.params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit(self, X, y, epochs: int = 300, lr: float = 0.01, batch: int = 32, optimizer: str = "adam",
            l2: float = 1e-4, seed: int = 0) -> list[float]:
        rng = np.random.default_rng(seed)
        history = []
        for _ in range(epochs):
            idx = rng.permutation(len(X))                 # mini-batchs mélangés à chaque époque
            for i in range(0, len(X), batch):
                b = idx[i:i + batch]
                self.step(self.gradients(X[b], y[b], l2), lr, optimizer)
            history.append(self.loss(X, y, l2))
        return history

    def predict(self, X):
        return self.forward(X)[0].argmax(1)


def gradient_check(model: MLP, X, y, eps: float = 1e-5, n_checks: int = 10, seed: int = 0) -> float:
    """Erreur relative max entre gradient analytique et numérique (doit être < 1e-5)."""
    rng = np.random.default_rng(seed)
    analytic = model.gradients(X, y)
    worst = 0.0
    for k, W in model.params.items():
        for _ in range(n_checks):
            idx = tuple(rng.integers(s) for s in W.shape)
            old = W[idx]
            W[idx] = old + eps
            lp = model.loss(X, y)
            W[idx] = old - eps
            lm = model.loss(X, y)
            W[idx] = old
            num = (lp - lm) / (2 * eps)
            ana = analytic[k][idx]
            worst = max(worst, abs(num - ana) / max(abs(num) + abs(ana), 1e-12))
    return worst


def spirals(n: int = 100, classes: int = 3, seed: int = 0):
    """Jeu de données classique NON linéairement séparable."""
    rng = np.random.default_rng(seed)
    X, y = [], []
    for c in range(classes):
        r = np.linspace(0.05, 1, n)
        t = np.linspace(c * 4, (c + 1) * 4, n) + rng.normal(0, 0.2, n)
        X.append(np.c_[r * np.sin(t), r * np.cos(t)])
        y += [c] * n
    return np.vstack(X), np.array(y)


def demo() -> None:
    X, y = spirals()
    m = MLP(2, 64, 3)
    print(f"Vérification du gradient : erreur relative max = {gradient_check(m, X[:20], y[:20]):.2e}")
    hist = m.fit(X, y, epochs=200, lr=0.01)
    print(f"Perte : {hist[0]:.3f} -> {hist[-1]:.3f} ; précision train = {np.mean(m.predict(X) == y):.2%}")
    print("(une régression logistique, linéaire, plafonnerait vers ~50 % sur ces spirales)")


if __name__ == "__main__":
    demo()
