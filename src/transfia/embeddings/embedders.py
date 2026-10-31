"""Embeddings (F4) : transformer un texte en vecteur.

Trois implémentations, de la plus simple à la plus puissante :

1. ``HashingEmbedder`` (FROM SCRATCH, aucun modèle) : n-grammes de caractères hachés dans un
   vecteur de taille fixe (le "hashing trick"). Capte la ressemblance de surface (orthographe,
   racines), PAS le sens. Ultra-rapide : une baseline frugale.
2. ``SentenceTransformerEmbedder`` : un encodeur transformer (ex. multilingual-e5-small) entraîné
   par apprentissage contrastif pour que deux textes de même SENS aient des vecteurs proches.
3. ``OllamaEmbedder`` : idem, mais servi par Ollama (ex. nomic-embed-text).

Tous renvoient des vecteurs NORMALISÉS (norme 1) : le produit scalaire = la similarité cosinus.
"""
from __future__ import annotations

import hashlib
import zlib

import numpy as np

from transfia.core.text import normalize
from transfia.observability.tracing import span


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-12)


class HashingEmbedder:
    def __init__(self, dim: int = 512, ngram_range: tuple[int, int] = (3, 5)):
        self.dim = dim
        self.ngram_range = ngram_range
        self.name = f"hashing-{dim}"

    def _features(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        t = f" {normalize(text)} "
        lo, hi = self.ngram_range
        for n in range(lo, hi + 1):
            for i in range(len(t) - n + 1):
                gram = t[i:i + n]
                h = zlib.crc32(gram.encode())
                # le signe (bit de poids fort) réduit l'effet des collisions de hachage
                v[h % self.dim] += 1.0 if (h >> 31) & 1 else -1.0
        # sous-linéarité : un n-gramme répété 10 fois ne compte pas 10 fois plus
        return np.sign(v) * np.log1p(np.abs(v))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return l2_normalize(np.stack([self._features(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


class SentenceTransformerEmbedder:
    """⚠️ Les modèles e5 attendent les préfixes "query: " et "passage: ". Les oublier dégrade la qualité."""

    def __init__(self, model: str = "intfloat/multilingual-e5-small", query_prefix: str = "query: ",
                 passage_prefix: str = "passage: ", batch_size: int = 16, path: str | None = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError('pip install -e ".[local]" (sentence-transformers)') from exc
        self._model = SentenceTransformer(path or model, device="cpu")    # path = copie locale dans models/
        self.name = model
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = get_dim()                               # nom de méthode changé selon la version
        self.query_prefix = query_prefix if "e5" in model.lower() else ""
        self.passage_prefix = passage_prefix if "e5" in model.lower() else ""
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        with span("embed.documents", model=self.name, n=len(texts)):
            x = self._model.encode([self.passage_prefix + t for t in texts], batch_size=self.batch_size,
                                   normalize_embeddings=True, show_progress_bar=len(texts) > 64)
        return np.asarray(x, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        x = self._model.encode([self.query_prefix + text], normalize_embeddings=True)
        return np.asarray(x[0], dtype=np.float32)


class OllamaEmbedder:
    def __init__(self, client, model: str = "nomic-embed-text"):
        self.client = client
        self.name = f"ollama/{model}"
        self.model = model
        self.dim = len(self.client.embed(["test"], model)[0])

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 32):
            out.extend(self.client.embed(texts[i:i + 32], self.model))
        return l2_normalize(np.asarray(out, dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_documents([text])[0]


def build_embedder(settings):
    cfg = settings.embedder
    if cfg.kind == "hashing":
        return HashingEmbedder(cfg.dim)
    if cfg.kind == "ollama":
        from transfia.llm.ollama import OllamaClient

        return OllamaEmbedder(OllamaClient(base_url=settings.llm.base_url), cfg.model)
    return SentenceTransformerEmbedder(cfg.model, cfg.query_prefix, cfg.passage_prefix,
                                       path=settings.model_path(cfg.model))


def text_fingerprint(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:10]
