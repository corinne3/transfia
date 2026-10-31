"""Pipeline d'ingestion (F1 -> F3 -> F4) : catalogue -> parsing -> chunking -> NER -> embeddings -> index.

Batch, IDEMPOTENT et INCRÉMENTAL : un manifeste mémorise les articles déjà indexés et la
configuration utilisée. Changer de modèle d'embedding impose de tout ré-indexer (les espaces
vectoriels de deux modèles ne sont pas compatibles) : le manifeste le détecte.
"""
from __future__ import annotations

import json
from pathlib import Path

from transfia.core.models import Article, Chunk, ParsedDocument, Section
from transfia.ingestion.chunking import chunk_document
from transfia.ingestion.ner import GazetteerNER, summarize_entities
from transfia.ingestion.themes import weak_label, weak_labels
from transfia.observability.metrics import EnergyMeter
from transfia.observability.tracing import span
from transfia.retrieval.bm25 import BM25


class Indexer:
    def __init__(self, settings, embedder, store, parser=None, ner=None, log=print):
        self.settings = settings
        self.embedder = embedder
        self.store = store
        self.parser = parser
        self.ner = ner or GazetteerNER()
        self.log = log
        self.manifest_path = settings.path("index", "manifest.json")
        self.bm25_path = settings.path("index", "bm25.json")
        self.manifest = self._load_manifest()

    # -------------------------------------------------------------- manifeste
    def _signature(self) -> dict:
        ing = self.settings.ingestion
        return {"embedder": self.embedder.name, "chunking": ing.chunking, "chunk_words": ing.chunk_words,
                "parser": ing.parser}

    def _load_manifest(self) -> dict:
        if self.manifest_path.exists():
            m = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if m.get("signature") != self._signature():
                self.log("⚠️  Configuration d'indexation différente de l'index existant : "
                         "supprime data/index/ pour ré-indexer proprement.")
            return m
        return {"signature": self._signature(), "articles": {}}

    def _save(self, all_chunks_for_bm25: list[Chunk]) -> None:
        self.store.save()
        BM25().fit([c.id for c in all_chunks_for_bm25], [c.text for c in all_chunks_for_bm25]).save(self.bm25_path)
        self.manifest["signature"] = self._signature()
        self.manifest_path.write_text(json.dumps(self.manifest, indent=1, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------- traitement
    def enrich(self, chunks: list[Chunk], article: Article) -> None:
        doc_text = f"{article.title} {article.abstract} {' '.join(article.keywords)}"
        art_theme = weak_label(doc_text + " " + " ".join(c.text for c in chunks[:5]))
        for c in chunks:
            ents = summarize_entities(self.ner.extract(c.text))
            c.meta["entities"] = ents
            c.meta["species"] = ents.get("espece", [])
            c.meta["themes"] = weak_labels(c.text, 1) or ([art_theme] if art_theme else [])
            c.meta["article_theme"] = art_theme

    def index_parsed(self, article: Article, parsed: ParsedDocument) -> list[Chunk]:
        with span("ingest.article", article=article.id) as s:
            if article.abstract:
                parsed.sections.insert(0, Section(title="Résumé", text=article.abstract))
            ing = self.settings.ingestion
            cstats: dict = {}
            chunks = chunk_document(parsed, article, ing.chunking, ing.chunk_words, ing.overlap_words, self.embedder,
                                    stats=cstats)
            self.enrich(chunks, article)
            if hasattr(self.store, "delete_article"):     # ré-indexation : pas de vieux chunks orphelins
                self.store.delete_article(article.id)
            if chunks:
                vectors = self.embedder.embed_documents([c.text for c in chunks])
                self.store.upsert(chunks, vectors)
            self.manifest["articles"][article.id] = {"n_chunks": len(chunks), "parser": parsed.parser,
                                                     "references_skipped": cstats.get("references_skipped", 0)}
            s.set(n_chunks=len(chunks))
            return chunks

    def run(self, items: list[tuple[Article, ParsedDocument | str | None]], force: bool = False) -> dict:
        """items : (article, document déjà parsé | chemin PDF | None = résumé seul)."""
        meter = EnergyMeter(self.settings.observability.energy, self.settings.observability.cpu_watts,
                            self.settings.observability.grid_gco2_per_kwh)
        stats = {"articles": 0, "skipped": 0, "chunks": 0, "errors": 0}
        with meter.measure("ingestion") as m:
            for article, source in items:
                if not force and article.id in self.manifest["articles"]:
                    stats["skipped"] += 1
                    continue
                try:
                    if isinstance(source, ParsedDocument):
                        parsed = source
                    elif isinstance(source, (str, Path)) and self.parser is not None:
                        parsed = self.parser.parse(str(source), article.id)
                    else:
                        parsed = ParsedDocument(article_id=article.id, parser="abstract")
                    chunks = self.index_parsed(article, parsed)
                    stats["articles"] += 1
                    stats["chunks"] += len(chunks)
                    refs = self.manifest["articles"][article.id].get("references_skipped", 0)
                    self.log(f"  ✓ {article.id} : {len(chunks)} chunks"
                             + (f" ({refs} passages de bibliographie écartés)" if refs else ""))
                    stats["references_skipped"] = stats.get("references_skipped", 0) + refs
                except Exception as exc:  # un PDF corrompu ne doit pas arrêter tout le batch
                    stats["errors"] += 1
                    self.log(f"  ✗ {article.id} : {type(exc).__name__}: {exc}")
            self._save(self.store.all_chunks())
        stats["energy"] = m.as_dict()
        stats["total_chunks"] = self.store.count()
        return stats


def load_bm25(settings) -> BM25:
    p = settings.path("index", "bm25.json")
    return BM25.load(p) if p.exists() else BM25()
