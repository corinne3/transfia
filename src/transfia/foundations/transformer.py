"""Le transformer décodé pièce par pièce (NumPy) : ce qu'il y a DANS un LLM.

1. TOKENISATION BPE (Byte Pair Encoding) : on part des caractères et on fusionne de façon répétée
   la paire de symboles la plus fréquente. Les mots fréquents deviennent un seul token, les mots
   rares sont découpés en morceaux. (~1,3 token par mot en français.)
2. EMBEDDINGS + ENCODAGE DE POSITION : chaque token -> un vecteur ; on y ajoute la position
   (sinus/cosinus de fréquences différentes), car l'attention seule ignore l'ordre.
3. SELF-ATTENTION : chaque token construit une requête Q, une clé K et une valeur V.
       Attention(Q, K, V) = softmax(Q Kᵀ / √d_k + masque) V
   Le score Q·K dit "à quel point le token j est utile au token i". Division par √d_k : sinon
   les scores grandissent avec la dimension et le softmax sature. MASQUE CAUSAL : un token ne
   voit pas le futur (génération de gauche à droite).
4. MULTI-HEAD : plusieurs attentions en parallèle (chaque "tête" apprend un type de relation).
5. BLOC TRANSFORMER : x + Attention(LayerNorm(x)), puis x + MLP(LayerNorm(x)) (connexions résiduelles).
6. GÉNÉRATION : logits -> probabilités -> échantillonnage (glouton, température, top-k, top-p).
   Le KV CACHE (section 11.1) mémorise K et V des tokens déjà vus pour ne pas les recalculer.

Les poids ici sont ALÉATOIRES (on montre les calculs, pas un modèle entraîné). Pour voir une
génération qui a du sens, ``BigramLM`` apprend par simple comptage.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


# ---------------------------------------------------------------- tokenisation
class BPETokenizer:
    def __init__(self):
        self.merges: list[tuple[str, str]] = []
        self.vocab: dict[str, int] = {}

    @staticmethod
    def _words(text: str) -> Counter:
        return Counter(tuple(w) + ("</w>",) for w in text.lower().split())

    def train(self, text: str, n_merges: int = 100) -> "BPETokenizer":
        words = self._words(text)
        for _ in range(n_merges):
            pairs = Counter()
            for w, f in words.items():
                for a, b in zip(w, w[1:]):
                    pairs[(a, b)] += f
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            self.merges.append(best)
            new_words = Counter()
            for w, f in words.items():
                out, i = [], 0
                while i < len(w):
                    if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                        out.append(w[i] + w[i + 1])
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                new_words[tuple(out)] += f
            words = new_words
        symbols = sorted({s for w in words for s in w} | {c for c in text.lower() if not c.isspace()} | {"</w>"})
        self.vocab = {s: i for i, s in enumerate(symbols)}
        return self

    def tokenize(self, text: str) -> list[str]:
        out = []
        for word in text.lower().split():
            w = list(word) + ["</w>"]
            for a, b in self.merges:                    # on rejoue les fusions dans l'ordre appris
                i, merged = 0, []
                while i < len(w):
                    if i < len(w) - 1 and w[i] == a and w[i + 1] == b:
                        merged.append(a + b)
                        i += 2
                    else:
                        merged.append(w[i])
                        i += 1
                w = merged
            out.extend(w)
        return out

    def encode(self, text: str) -> list[int]:
        return [self.vocab.get(t, self.vocab.get("</w>", 0)) for t in self.tokenize(text)]

    def decode(self, ids: list[int]) -> str:
        inv = {i: s for s, i in self.vocab.items()}
        return "".join(inv[i] for i in ids).replace("</w>", " ").strip()


# ------------------------------------------------------------------- briques
def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def positional_encoding(seq_len: int, d: int) -> np.ndarray:
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d)
    return np.where(i % 2 == 0, np.sin(angle), np.cos(angle))


def causal_mask(n: int) -> np.ndarray:
    return np.triu(np.full((n, n), -np.inf), k=1)          # -inf au-dessus de la diagonale


def scaled_dot_product_attention(Q, K, V, mask=None):
    scores = Q @ K.swapaxes(-1, -2) / np.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores + mask
    weights = softmax(scores, axis=-1)
    return weights @ V, weights


def layer_norm(x, eps: float = 1e-5):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps)


class MultiHeadAttention:
    def __init__(self, d_model: int, n_heads: int, rng):
        assert d_model % n_heads == 0
        self.h, self.dk = n_heads, d_model // n_heads
        s = 1 / np.sqrt(d_model)
        self.Wq, self.Wk, self.Wv, self.Wo = (rng.normal(0, s, (d_model, d_model)) for _ in range(4))

    def __call__(self, x, mask=None):
        n, d = x.shape

        def split(t):                                     # (n, d) -> (têtes, n, dk)
            return t.reshape(n, self.h, self.dk).transpose(1, 0, 2)

        Q, K, V = split(x @ self.Wq), split(x @ self.Wk), split(x @ self.Wv)
        out, weights = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 0, 2).reshape(n, d)        # concaténation des têtes
        return out @ self.Wo, weights


class TransformerBlock:
    def __init__(self, d_model: int = 32, n_heads: int = 4, d_ff: int = 64, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.attn = MultiHeadAttention(d_model, n_heads, rng)
        self.W1 = rng.normal(0, 1 / np.sqrt(d_model), (d_model, d_ff))
        self.W2 = rng.normal(0, 1 / np.sqrt(d_ff), (d_ff, d_model))

    def __call__(self, x, causal: bool = True):
        a, weights = self.attn(layer_norm(x), causal_mask(len(x)) if causal else None)
        x = x + a                                         # connexion résiduelle
        h = np.maximum(layer_norm(x) @ self.W1, 0) @ self.W2
        return x + h, weights


class TinyTransformerLM:
    """Embeddings -> blocs -> projection sur le vocabulaire (poids aléatoires : démonstration des formes)."""

    def __init__(self, vocab_size: int, d_model: int = 32, n_layers: int = 2, n_heads: int = 4, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.emb = rng.normal(0, 0.1, (vocab_size, d_model))
        self.blocks = [TransformerBlock(d_model, n_heads, 2 * d_model, seed + i) for i in range(n_layers)]

    def logits(self, ids: list[int]) -> np.ndarray:
        x = self.emb[ids] + positional_encoding(len(ids), self.emb.shape[1])
        for b in self.blocks:
            x, _ = b(x)
        return layer_norm(x) @ self.emb.T                 # "weight tying" : on réutilise la matrice d'embedding

    def n_params(self) -> int:
        total = self.emb.size
        for b in self.blocks:
            total += sum(w.size for w in (b.attn.Wq, b.attn.Wk, b.attn.Wv, b.attn.Wo, b.W1, b.W2))
        return total


# -------------------------------------------------------------- échantillonnage
def sample_next(logits: np.ndarray, temperature: float = 1.0, top_k: int | None = None, top_p: float | None = None,
                rng=None) -> int:
    """temperature -> 0 : glouton (toujours le plus probable) ; > 1 : plus aléatoire.
    top-k : on ne garde que les k tokens les plus probables.
    top-p (nucleus) : on garde le plus petit ensemble dont la probabilité cumulée dépasse p."""
    rng = rng or np.random.default_rng()
    if temperature <= 1e-6:
        return int(np.argmax(logits))
    probs = softmax(logits / temperature)
    if top_k is not None:
        cutoff = np.sort(probs)[-top_k]
        probs = np.where(probs >= cutoff, probs, 0)
    if top_p is not None:
        order = np.argsort(-probs)
        cum = np.cumsum(probs[order])
        keep = order[: int(np.searchsorted(cum, top_p * cum[-1])) + 1]
        mask = np.zeros_like(probs)
        mask[keep] = 1
        probs = probs * mask
    probs = probs / probs.sum()
    return int(rng.choice(len(probs), p=probs))


class BigramLM:
    """Le plus petit "modèle de langage" : P(mot suivant | mot courant), appris par COMPTAGE.
    Un LLM fait la même chose (prédire le token suivant), avec un contexte de milliers de tokens."""

    def fit(self, text: str) -> "BigramLM":
        words = text.lower().split()
        self.vocab = sorted(set(words))
        self.idx = {w: i for i, w in enumerate(self.vocab)}
        V = len(self.vocab)
        self.counts = np.ones((V, V))                     # lissage +1
        for a, b in zip(words, words[1:]):
            self.counts[self.idx[a], self.idx[b]] += 1
        return self

    def generate(self, start: str, n: int = 10, temperature: float = 0.8, seed: int = 0) -> str:
        rng = np.random.default_rng(seed)
        w = start.lower()
        out = [w]
        for _ in range(n):
            if w not in self.idx:
                break
            logits = np.log(self.counts[self.idx[w]])
            w = self.vocab[sample_next(logits, temperature, top_k=5, rng=rng)]
            out.append(w)
        return " ".join(out)

    def perplexity(self, text: str) -> float:
        """exp(entropie croisée moyenne) : « entre combien de mots le modèle hésite en moyenne »."""
        words = [w for w in text.lower().split() if w in self.idx]
        probs = self.counts / self.counts.sum(1, keepdims=True)
        nll = [-np.log(probs[self.idx[a], self.idx[b]]) for a, b in zip(words, words[1:])]
        return float(np.exp(np.mean(nll))) if nll else float("inf")


CORPUS = ("la vache produit du lait . la vache mange du fourrage . le veau boit le colostrum . "
          "le colostrum protège le veau . la brebis mange de l'herbe au pâturage . le pâturage nourrit la brebis . "
          "la truie allaite le porcelet . le porcelet mange un aliment après le sevrage .")


def demo() -> None:
    tok = BPETokenizer().train(CORPUS * 3, n_merges=40)
    print("BPE :", tok.tokenize("le colostrum protège les veaux"))
    ids = tok.encode("la vache mange")
    print("ids :", ids, "->", tok.decode(ids))
    x = np.random.default_rng(0).normal(size=(5, 16))
    _, w = scaled_dot_product_attention(x, x, x, causal_mask(5))
    print("Poids d'attention causale (ligne i = ce que regarde le token i) :\n", w.round(2))
    lm = TinyTransformerLM(len(tok.vocab))
    print(f"Mini-transformer : {lm.n_params():,} paramètres ; logits pour 3 tokens -> forme {lm.logits(ids).shape}")
    bigram = BigramLM().fit(CORPUS)
    print("Bigramme, température 0.8 :", bigram.generate("le", 8))
    print("Bigramme, perplexité sur le corpus :", round(bigram.perplexity(CORPUS), 2))


if __name__ == "__main__":
    demo()
