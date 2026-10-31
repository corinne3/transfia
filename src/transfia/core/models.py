"""Modèles de données partagés (Pydantic).

Pydantic sert ici à deux choses :
1. typer et valider les données qui circulent entre les briques ;
2. produire des JSON schema pour les sorties structurées des LLM (F7).
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class Article(BaseModel):
    """Une entrée du catalogue (F1). L'identifiant stable est le DOI quand il existe."""

    id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    abstract: str = ""
    keywords: list[str] = Field(default_factory=list)
    license: str | None = None
    pdf_url: str | None = None
    landing_url: str | None = None
    source: str = "openalex"

    def citation_label(self) -> str:
        first = self.authors[0].split()[-1] if self.authors else "Anonyme"
        etal = " et al." if len(self.authors) > 1 else ""
        return f"{first}{etal}, {self.year or 's.d.'}"


class Figure(BaseModel):
    page: int
    caption: str = ""
    description: str = ""          # générée par un VLM : marquée comme telle
    generated: bool = False


class Section(BaseModel):
    title: str
    text: str
    page: int | None = None


class ParsedDocument(BaseModel):
    """Sortie du parsing (F2) : un document structuré, pas juste du texte."""

    article_id: str
    sections: list[Section] = Field(default_factory=list)
    tables: list[str] = Field(default_factory=list)        # en Markdown
    figures: list[Figure] = Field(default_factory=list)
    n_pages: int = 0
    parser: str = ""

    @property
    def full_text(self) -> str:
        return "\n\n".join(f"{s.title}\n{s.text}" for s in self.sections)


class Chunk(BaseModel):
    """Un morceau de document indexé. Il garde TOUJOURS ses métadonnées de provenance."""

    id: str
    article_id: str
    text: str
    section: str = ""
    page: int | None = None
    meta: dict[str, Any] = Field(default_factory=dict)     # titre, auteurs, année, DOI, entités...


class Hit(BaseModel):
    chunk: Chunk
    score: float
    source: str = ""               # "dense", "bm25", "hybrid", "rerank"
    rank: int = 0


class Citation(BaseModel):
    n: int                         # numéro du passage dans le prompt [n]
    article_id: str
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    page: int | None = None
    excerpt: str = ""

    @property
    def doi_url(self) -> str | None:
        return f"https://doi.org/{self.doi}" if self.doi else None


class SentenceCheck(BaseModel):
    """Vérification d'une phrase de la réponse contre ses sources (F5)."""

    sentence: str
    cited: list[int] = Field(default_factory=list)
    supported: bool
    score: float
    method: str = "lexical"
    bad_citations: list[int] = Field(default_factory=list)   # passages cités qui ne soutiennent PAS la phrase


class Answer(BaseModel):
    question: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    checks: list[SentenceCheck] = Field(default_factory=list)
    refused: bool = False
    refusal_reason: str | None = None
    queries: list[str] = Field(default_factory=list)
    hits: list[Hit] = Field(default_factory=list)
    trace_id: str | None = None
    model: str | None = None
    usage: dict[str, Any] = Field(default_factory=dict)

    @property
    def faithfulness(self) -> float | None:
        if not self.checks:
            return None
        return sum(c.supported for c in self.checks) / len(self.checks)
