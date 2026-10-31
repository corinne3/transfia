"""Machine learning classique codé from scratch (NumPy). Idéal pour réviser un test technique.

Contenu :
- découpage train/test, validation croisée k-fold, métriques (précision, rappel, F1, MSE, R²)
- régression linéaire par DESCENTE DE GRADIENT (et solution exacte pour vérifier)
- régression logistique (sigmoïde, entropie croisée, régularisation L2)
- k plus proches voisins (kNN)
- Naive Bayes multinomial (probabilités + lissage de Laplace) pour du texte
- arbre de décision (impureté de Gini)
- KMeans (initialisation k-means++)
- PCA (par décomposition en valeurs singulières)

Notions clés :
- BIAIS / VARIANCE : un modèle trop simple sous-apprend (biais), trop complexe sur-apprend (variance).
- RÉGULARISATION : pénaliser les grands poids (L2) réduit la variance.
- Toujours évaluer sur des données NON VUES pendant l'entraînement.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


# ------------------------------------------------------------------ outils
def train_test_split(X, y, test_size: float = 0.25, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(len(X) * test_size)
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], X[te], y[tr], y[te]


def kfold_indices(n: int, k: int = 5, seed: int = 0):
    idx = np.random.default_rng(seed).permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        test = folds[i]
        train = np.concatenate([folds[j] for j in range(k) if j != i])
        yield train, test


def precision_recall_f1(y_true, y_pred, positive=1) -> tuple[float, float, float]:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tp = np.sum((y_pred == positive) & (y_true == positive))
    fp = np.sum((y_pred == positive) & (y_true != positive))
    fn = np.sum((y_pred != positive) & (y_true == positive))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return float(p), float(r), float(f1)


def mse(y, yhat) -> float:
    return float(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2))


def r2(y, yhat) -> float:
    y, yhat = np.asarray(y), np.asarray(yhat)
    return float(1 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2))


def standardize(X_train, X_test):
    """Moyenne/écart-type calculés sur le TRAIN seulement (sinon : fuite d'information)."""
    mu, sd = X_train.mean(axis=0), X_train.std(axis=0) + 1e-12
    return (X_train - mu) / sd, (X_test - mu) / sd


# ------------------------------------------------------- régression linéaire
class LinearRegressionGD:
    """Minimise MSE = 1/n Σ (Xw + b − y)². Gradient : dw = 2/n Xᵀ(Xw + b − y), db = 2/n Σ(Xw + b − y)."""

    def __init__(self, lr: float = 0.05, epochs: int = 2000, l2: float = 0.0):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y):
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        self.losses = []
        for _ in range(self.epochs):
            err = X @ self.w + self.b - y
            self.w -= self.lr * (2 / n * X.T @ err + 2 * self.l2 * self.w)
            self.b -= self.lr * 2 / n * err.sum()
            self.losses.append(float(np.mean(err ** 2)))
        return self

    def predict(self, X):
        return X @ self.w + self.b


def normal_equation(X, y):
    """Solution exacte (moindres carrés) pour vérifier la descente de gradient."""
    Xb = np.hstack([X, np.ones((len(X), 1))])
    theta = np.linalg.lstsq(Xb, y, rcond=None)[0]
    return theta[:-1], theta[-1]


# ----------------------------------------------------- régression logistique
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


class LogisticRegressionGD:
    """p = σ(Xw + b). Perte = entropie croisée binaire −[y log p + (1−y) log(1−p)] + λ‖w‖².
    Gradient (élégant !) : dw = 1/n Xᵀ(p − y) + 2λw."""

    def __init__(self, lr: float = 0.1, epochs: int = 3000, l2: float = 0.01):
        self.lr, self.epochs, self.l2 = lr, epochs, l2

    def fit(self, X, y):
        n, d = X.shape
        self.w, self.b = np.zeros(d), 0.0
        for _ in range(self.epochs):
            p = sigmoid(X @ self.w + self.b)
            self.w -= self.lr * (X.T @ (p - y) / n + 2 * self.l2 * self.w)
            self.b -= self.lr * float(np.mean(p - y))
        return self

    def predict_proba(self, X):
        return sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold: float = 0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


# ------------------------------------------------------------------- kNN
class KNN:
    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, X, y):
        self.X, self.y = X, y
        return self

    def predict(self, X):
        d = ((X[:, None, :] - self.X[None, :, :]) ** 2).sum(-1)       # distances au carré (n_test, n_train)
        nn = np.argsort(d, axis=1)[:, : self.k]
        return np.array([Counter(self.y[row]).most_common(1)[0][0] for row in nn])


# ----------------------------------------------------------- Naive Bayes
class MultinomialNB:
    """P(classe | mots) ∝ P(classe) × Π P(mot | classe)  (hypothèse "naïve" : mots indépendants).
    On travaille en LOG pour éviter de multiplier des milliers de petites probabilités (underflow).
    Lissage de Laplace (+α) : un mot jamais vu dans une classe n'annule pas tout."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, docs: list[list[str]], y: list[str]):
        self.classes = sorted(set(y))
        self.vocab = sorted({w for d in docs for w in d})
        self.log_prior, self.log_lik = {}, {}
        for c in self.classes:
            cdocs = [d for d, lab in zip(docs, y) if lab == c]
            self.log_prior[c] = np.log(len(cdocs) / len(docs))
            counts = Counter(w for d in cdocs for w in d)
            total = sum(counts.values()) + self.alpha * len(self.vocab)
            self.log_lik[c] = {w: np.log((counts[w] + self.alpha) / total) for w in self.vocab}
            self.log_lik[c]["<inconnu>"] = np.log(self.alpha / total)
        return self

    def predict(self, docs: list[list[str]]) -> list[str]:
        out = []
        for d in docs:
            scores = {c: self.log_prior[c] + sum(self.log_lik[c].get(w, self.log_lik[c]["<inconnu>"]) for w in d)
                      for c in self.classes}
            out.append(max(scores, key=scores.get))
        return out


# ------------------------------------------------------- arbre de décision
def gini(y) -> float:
    """Impureté de Gini = 1 − Σ p_k² (0 = nœud pur)."""
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return float(1 - np.sum(p ** 2))


class DecisionTree:
    def __init__(self, max_depth: int = 4, min_samples: int = 2):
        self.max_depth, self.min_samples = max_depth, min_samples

    def fit(self, X, y):
        self.tree = self._build(X, y, 0)
        return self

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples or gini(y) == 0:
            return Counter(y.tolist()).most_common(1)[0][0]
        best = None
        for j in range(X.shape[1]):
            for t in np.unique(X[:, j]):
                left = X[:, j] <= t
                if left.all() or (~left).all():
                    continue
                # gain = impureté parent − impureté pondérée des enfants
                score = (left.sum() * gini(y[left]) + (~left).sum() * gini(y[~left])) / len(y)
                if best is None or score < best[0]:
                    best = (score, j, t, left)
        if best is None:
            return Counter(y.tolist()).most_common(1)[0][0]
        _, j, t, left = best
        return (j, t, self._build(X[left], y[left], depth + 1), self._build(X[~left], y[~left], depth + 1))

    def _predict_one(self, x, node):
        while isinstance(node, tuple):
            j, t, l, r = node
            node = l if x[j] <= t else r
        return node

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])


# ---------------------------------------------------------------- KMeans
def kmeans(X, k: int, iters: int = 100, seed: int = 0):
    """Alterner : (1) affecter chaque point au centre le plus proche, (2) recalculer les centres.
    Initialisation k-means++ : centres éloignés les uns des autres -> convergence meilleure."""
    rng = np.random.default_rng(seed)
    centers = [X[rng.integers(len(X))]]
    for _ in range(1, k):
        d2 = np.min([((X - c) ** 2).sum(1) for c in centers], axis=0)
        centers.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    centers = np.array(centers)
    for _ in range(iters):
        labels = np.argmin(((X[:, None] - centers[None]) ** 2).sum(-1), axis=1)
        new = np.array([X[labels == j].mean(0) if np.any(labels == j) else centers[j] for j in range(k)])
        if np.allclose(new, centers):
            break
        centers = new
    inertia = float(((X - centers[labels]) ** 2).sum())
    return labels, centers, inertia


# ------------------------------------------------------------------- PCA
def pca(X, n_components: int = 2):
    """Centrer, puis SVD : X = U S Vᵀ ; les lignes de Vᵀ sont les axes principaux.
    Variance expliquée par l'axe i = S_i² / Σ S²."""
    Xc = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    explained = S ** 2 / np.sum(S ** 2)
    return Xc @ Vt[:n_components].T, explained[:n_components]


def demo() -> None:
    rng = np.random.default_rng(0)
    # régression : poids de veaux (kg) en fonction de l'âge (jours) — données SIMULÉES
    age = rng.uniform(0, 120, 200)
    poids = 40 + 0.8 * age + rng.normal(0, 6, 200)
    X = age[:, None]
    Xtr, Xte, ytr, yte = train_test_split(X, poids)
    Xtr_s, Xte_s = standardize(Xtr, Xte)
    lr = LinearRegressionGD().fit(Xtr_s, ytr)
    print(f"Régression linéaire (GD) : R² test = {r2(yte, lr.predict(Xte_s)):.3f}")
    # classification : deux nuages de points
    A = rng.normal([0, 0], 1, (100, 2))
    B = rng.normal([2.5, 2.5], 1, (100, 2))
    Xc = np.vstack([A, B])
    yc = np.array([0] * 100 + [1] * 100)
    Xtr, Xte, ytr, yte = train_test_split(Xc, yc)
    for name, m in [("logistique", LogisticRegressionGD()), ("kNN (k=5)", KNN(5)), ("arbre (prof. 3)", DecisionTree(3))]:
        p, r, f = precision_recall_f1(yte, m.fit(Xtr, ytr).predict(Xte))
        print(f"{name:16s} précision {p:.2f}  rappel {r:.2f}  F1 {f:.2f}")
    labels, _, inertia = kmeans(Xc, 2)
    print(f"KMeans : inertie {inertia:.1f}, tailles des groupes {np.bincount(labels).tolist()}")
    _, explained = pca(np.hstack([Xc, Xc @ np.array([[1.0], [0.5]])]), 2)
    print(f"PCA : variance expliquée par les 2 premiers axes = {explained.round(3).tolist()}")
    nb = MultinomialNB().fit([["mammite", "lait"], ["methane", "rumen"], ["lait", "cellules"], ["emission", "methane"]],
                             ["sante", "environnement", "sante", "environnement"])
    print("Naive Bayes :", nb.predict([["methane", "emission"], ["lait", "mammite"]]))


if __name__ == "__main__":
    demo()
