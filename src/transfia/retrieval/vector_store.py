"""Bases vectorielles (F4).

1. ``InMemoryVectorStore`` (from scratch) : recherche EXACTE par produit matriciel NumPy.
   Parfait jusqu'à ~100 000 vecteurs sur un PC. Persistance : .npy + .jsonl.
2. ``QdrantVectorStore`` : Qdrant en mode local (un dossier, sans serveur). Au-delà de quelques
   milliers de vecteurs, Qdrant utilise HNSW, un index ANN (approximate nearest neighbors) :
   un graphe multi-niveaux de voisins qu'on parcourt du haut vers le bas, au lieu de comparer
   la requête à TOUS les vecteurs. On échange un peu de rappel contre beaucoup de vitesse.

Les deux gèrent des FILTRES sur les métadonnées (espèce, thème, année...).
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import numpy as np

from transfia.core.models import Chunk, Hit
from transfia.retrieval.similarity import top_k


def _match(meta: dict, filters: dict | None) -> bool:
    """Filtre simple : égalité, ou appartenance si la valeur du filtre/de la méta est une liste.

    Clés spéciales : "year_min", "year_max".
    """
    if not filters:
        return True
    for key, want in filters.items():
        if want in (None, "", []):
            continue
        if key == "year_min":
            if (meta.get("year") or 0) < int(want):
                return False
            continue
        if key == "year_max":
            if (meta.get("year") or 9999) > int(want):
                return False
            continue
        have = meta.get(key)
        wants = want if isinstance(want, list) else [want]
        haves = have if isinstance(have, list) else [have]
        if not set(map(str, wants)) & set(map(str, haves)):
            return False
    return True


class InMemoryVectorStore:
    def __init__(self, directory: Path | None = None):
        self.directory = directory
        self.chunks: list[Chunk] = []
        self.vectors: np.ndarray | None = None
        self._index: dict[str, int] = {}
        if directory and (directory / "vectors.npy").exists():
            self._load()

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        for c, v in zip(chunks, vectors):
            if c.id in self._index:                       # mise à jour
                i = self._index[c.id]
                self.chunks[i] = c
                self.vectors[i] = v
            else:                                         # ajout
                self._index[c.id] = len(self.chunks)
                self.chunks.append(c)
                self.vectors = v[None, :] if self.vectors is None else np.vstack([self.vectors, v])

    def search(self, vector: np.ndarray, k: int, filters: dict | None = None) -> list[Hit]:
        if self.vectors is None or not len(self.chunks):
            return []
        scores = self.vectors @ np.asarray(vector, dtype=np.float32)
        if filters:
            mask = np.array([_match(c.meta, filters) for c in self.chunks])
            scores = np.where(mask, scores, -np.inf)
        idx = [i for i in top_k(scores, k) if np.isfinite(scores[i])]
        return [Hit(chunk=self.chunks[i], score=float(scores[i]), source="dense", rank=r)
                for r, i in enumerate(idx)]

    def delete_article(self, article_id: str) -> int:
        """Supprime tous les chunks d'un article (ré-indexation : sinon d'anciens chunks restent)."""
        keep = [i for i, c in enumerate(self.chunks) if c.article_id != article_id]
        removed = len(self.chunks) - len(keep)
        if removed:
            self.chunks = [self.chunks[i] for i in keep]
            self.vectors = self.vectors[keep] if keep else None
            self._index = {c.id: i for i, c in enumerate(self.chunks)}
        return removed

    def get(self, chunk_id: str) -> Chunk | None:
        i = self._index.get(chunk_id)
        return None if i is None else self.chunks[i]

    def all_chunks(self) -> list[Chunk]:
        return list(self.chunks)

    def count(self) -> int:
        return len(self.chunks)

    def save(self) -> None:
        if not self.directory or self.vectors is None:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        np.save(self.directory / "vectors.npy", self.vectors)
        with (self.directory / "chunks.jsonl").open("w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(c.model_dump_json() + "\n")

    def _load(self) -> None:
        self.vectors = np.load(self.directory / "vectors.npy")
        lines = (self.directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
        self.chunks = [Chunk.model_validate_json(line) for line in lines if line.strip()]
        self._index = {c.id: i for i, c in enumerate(self.chunks)}


class QdrantVectorStore:
    COLLECTION = "transfia_chunks"

    def __init__(self, directory: Path, dim):
        """``dim`` : un entier, ou une fonction qui le renvoie (appelée seulement si la collection
        n'existe pas encore : on évite ainsi de charger le modèle d'embedding pour rien)."""
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:
            raise ImportError('pip install -e ".[local]" (qdrant-client)') from exc
        self.models = models
        directory.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(directory))
        if not self.client.collection_exists(self.COLLECTION):
            dim = dim() if callable(dim) else dim
            self.client.create_collection(
                self.COLLECTION,
                vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
            )

    @staticmethod
    def _pid(chunk_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))   # Qdrant veut un UUID ou un entier

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        m = self.models
        points = [m.PointStruct(id=self._pid(c.id), vector=v.tolist(),
                                payload={"chunk": c.model_dump(), **{k: v2 for k, v2 in c.meta.items()
                                                                     if isinstance(v2, (str, int, float, list))}})
                  for c, v in zip(chunks, vectors)]
        for i in range(0, len(points), 256):
            self.client.upsert(self.COLLECTION, points=points[i:i + 256])

    def _filter(self, filters: dict | None):
        if not filters:
            return None
        m = self.models
        must = []
        for key, want in filters.items():
            if want in (None, "", []):
                continue
            if key == "year_min":
                must.append(m.FieldCondition(key="year", range=m.Range(gte=int(want))))
            elif key == "year_max":
                must.append(m.FieldCondition(key="year", range=m.Range(lte=int(want))))
            else:
                wants = want if isinstance(want, list) else [want]
                must.append(m.FieldCondition(key=key, match=m.MatchAny(any=wants)))
        return m.Filter(must=must) if must else None

    def search(self, vector: np.ndarray, k: int, filters: dict | None = None) -> list[Hit]:
        res = self.client.query_points(self.COLLECTION, query=np.asarray(vector).tolist(), limit=k,
                                       query_filter=self._filter(filters), with_payload=True).points
        return [Hit(chunk=Chunk(**p.payload["chunk"]), score=float(p.score), source="dense", rank=r)
                for r, p in enumerate(res)]

    def delete_article(self, article_id: str) -> int:
        ids = [self._pid(c.id) for c in self.all_chunks() if c.article_id == article_id]
        if ids:
            self.client.delete(self.COLLECTION, points_selector=self.models.PointIdsList(points=ids))
        return len(ids)

    def get(self, chunk_id: str) -> Chunk | None:
        res = self.client.retrieve(self.COLLECTION, ids=[self._pid(chunk_id)], with_payload=True)
        return Chunk(**res[0].payload["chunk"]) if res else None

    def all_chunks(self) -> list[Chunk]:
        out, offset = [], None
        while True:
            points, offset = self.client.scroll(self.COLLECTION, limit=512, offset=offset, with_payload=True)
            out.extend(Chunk(**p.payload["chunk"]) for p in points)
            if offset is None:
                return out

    def count(self) -> int:
        return self.client.count(self.COLLECTION).count

    def save(self) -> None:  # Qdrant local persiste tout seul
        pass


def build_vector_store(settings, dim):
    if settings.retrieval.vector_store == "qdrant":
        return QdrantVectorStore(settings.path("index", "qdrant", ".keep").parent, dim)
    return InMemoryVectorStore(settings.path("index", "memory", ".keep").parent)


def dump_chunks(chunks: list[Chunk], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")
