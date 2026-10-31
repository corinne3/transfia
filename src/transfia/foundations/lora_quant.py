"""LoRA et quantification, en NumPy : les deux techniques qui rendent les LLM « frugaux ».

LoRA (Low-Rank Adaptation, F10)
- Au lieu de modifier une grande matrice W (d × k), on apprend une correction de RANG r :
      W' = W + (α / r) · B A,   avec B (d × r), A (r × k), r ≪ d, k
- W reste GELÉE ; seuls A et B s'entraînent : ~1 % des paramètres.
- B est initialisée à ZÉRO : au départ, W' = W (le modèle n'est pas perturbé).
- Après l'entraînement, on peut FUSIONNER : W' = W + (α/r) B A (aucun surcoût à l'inférence).
- QLoRA = W chargée en 4 bits + adaptateurs LoRA en précision normale.

QUANTIFICATION (11.1)
- On stocke les poids sur moins de bits : 16 -> 8 -> 4 bits, avec une ÉCHELLE pour retrouver la valeur.
- Symétrique par ligne : q = round(w / s), s = max|w| / (2^(b-1) − 1) ; déquantification : w ≈ q · s.
- Par GROUPES (ex. 32 poids partagent une échelle) : une valeur extrême n'écrase pas la précision
  de toute la ligne -> c'est l'idée des formats GGUF « Q4_K_M » d'Ollama / llama.cpp.
"""
from __future__ import annotations

import numpy as np


class LoRALinear:
    def __init__(self, W: np.ndarray, r: int = 8, alpha: float = 16, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.W = W                                        # gelée
        d_out, d_in = W.shape
        self.A = rng.normal(0, 1 / np.sqrt(d_in), (r, d_in))
        self.B = np.zeros((d_out, r))
        self.scale = alpha / r

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return x @ self.W.T + self.scale * (x @ self.A.T) @ self.B.T

    def trainable_params(self) -> int:
        return self.A.size + self.B.size

    def merged(self) -> np.ndarray:
        return self.W + self.scale * self.B @ self.A

    def fit(self, X, Y, lr: float = 0.05, epochs: int = 300) -> list[float]:
        """Apprend A et B pour que la couche reproduise Y (descente de gradient sur la MSE)."""
        losses = []
        for _ in range(epochs):
            h = X @ self.A.T                              # (n, r)
            err = self(X) - Y                             # (n, d_out)
            gB = self.scale * err.T @ h / len(X)
            gA = self.scale * (err @ self.B).T @ X / len(X)
            self.B -= lr * gB
            self.A -= lr * gA
            losses.append(float(np.mean(err ** 2)))
        return losses


def quantize_symmetric(W: np.ndarray, bits: int = 8, group_size: int | None = None):
    qmax = 2 ** (bits - 1) - 1
    if group_size:
        flat = W.reshape(-1, group_size)
        scale = np.abs(flat).max(1, keepdims=True) / qmax + 1e-12
        q = np.clip(np.round(flat / scale), -qmax, qmax).astype(np.int8)
        return q, scale
    scale = np.abs(W).max(1, keepdims=True) / qmax + 1e-12
    q = np.clip(np.round(W / scale), -qmax, qmax).astype(np.int8)
    return q, scale


def dequantize(q: np.ndarray, scale: np.ndarray, shape: tuple | None = None) -> np.ndarray:
    w = q.astype(np.float32) * scale
    return w.reshape(shape) if shape else w


def quantization_error(W: np.ndarray, bits: int, group_size: int | None = None) -> float:
    q, s = quantize_symmetric(W, bits, group_size)
    return float(np.mean(np.abs(W - dequantize(q, s, W.shape))) / np.mean(np.abs(W)))


def memory_mb(n_params: int, bits: float) -> float:
    return round(n_params * bits / 8 / 1e6, 2)


def demo() -> None:
    rng = np.random.default_rng(0)
    d = 512
    W = rng.normal(0, 0.02, (d, d))
    W[0, 0] = 0.5                                         # une valeur extrême (fréquente dans les LLM)
    lora = LoRALinear(W, r=8)
    print(f"LoRA r=8 : {lora.trainable_params():,} paramètres entraînables sur {W.size:,} ({lora.trainable_params() / W.size:.1%})")
    # tâche : apprendre une petite correction de rang faible
    target_delta = rng.normal(0, 0.05, (d, 4)) @ rng.normal(0, 0.05, (4, d))
    X = rng.normal(size=(256, d))
    losses = lora.fit(X, X @ (W + target_delta).T, lr=0.01, epochs=300)   # lr trop grand = divergence !
    print(f"LoRA : MSE {losses[0]:.5f} -> {losses[-1]:.5f} ; fusion possible : W' de forme {lora.merged().shape}")
    for bits, g in [(8, None), (4, None), (4, 32)]:
        label = f"{bits} bits" + (f", groupes de {g}" if g else ", par ligne")
        print(f"Quantification {label:22s} erreur relative {quantization_error(W, bits, g):.3%}")
    n = 3_000_000_000
    print("Modèle de 3 Md de paramètres :", {f"{b} bits": f"{memory_mb(n, b) / 1000:.1f} Go" for b in (16, 8, 4)})


if __name__ == "__main__":
    demo()
