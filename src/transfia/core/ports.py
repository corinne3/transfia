"""Ports (interfaces) de l'architecture hexagonale (chapitre 12.3 du dossier).

Le code métier (RAG, agents, génération) ne dépend QUE de ces interfaces.
Chaque fournisseur (local, GCP, Azure...) fournit des adaptateurs qui les implémentent.
-> changer de cloud = écrire un adaptateur + changer la configuration.

``typing.Protocol`` = typage structurel : pas besoin d'hériter, il suffit d'avoir les bonnes méthodes
("duck typing" vérifiable par un outil comme mypy).
"""
from __future__ import annotations

from typing import Iterator, Protocol

import numpy as np

from transfia.core.models import Chunk, Hit, ParsedDocument


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None: ...
    def search(self, vector: np.ndarray, k: int, filters: dict | None = None) -> list[Hit]: ...
    def get(self, chunk_id: str) -> Chunk | None: ...
    def all_chunks(self) -> list[Chunk]: ...
    def count(self) -> int: ...
    def save(self) -> None: ...


class DocumentParser(Protocol):
    name: str

    def parse(self, path: str, article_id: str) -> ParsedDocument: ...


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]: ...


class FileStore(Protocol):
    def put(self, key: str, data: bytes) -> None: ...
    def get(self, key: str) -> bytes: ...
    def exists(self, key: str) -> bool: ...


class TokenStream(Protocol):
    def __iter__(self) -> Iterator[str]: ...
