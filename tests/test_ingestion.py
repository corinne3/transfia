"""F1-F3 — collecte (OpenAlex, OAI-PMH), parsing, chunking, NER, thèmes, pipeline idempotent."""
import json
import tempfile
from pathlib import Path

from helpers import make_ctx, require

from transfia.core.models import Article, ParsedDocument, Section
from transfia.core.text import split_sentences, tokenize
from transfia.embeddings.embedders import HashingEmbedder
from transfia.ingestion.chunking import chunk_document, fixed_size, semantic_split, sentence_pack
from transfia.ingestion.collect import (Catalog, article_from_openalex, is_reusable, iter_openalex,
                                        parse_oai_records, rebuild_abstract)
from transfia.ingestion.ner import GazetteerNER, summarize_entities
from transfia.ingestion.themes import weak_label

OPENALEX_WORK = {
    "id": "https://openalex.org/W123", "doi": "https://doi.org/10.20870/productions-animales.2024.37.1.0000",
    "title": "Une synthèse", "publication_year": 2024,
    "authorships": [{"author": {"display_name": "Marie Curie"}}, {"author": {"display_name": "Louis Pasteur"}}],
    "abstract_inverted_index": {"Le": [0], "colostrum": [1], "protège": [2], "le": [3], "veau": [4]},
    "best_oa_location": {"license": "cc-by", "pdf_url": "https://exemple.org/a.pdf", "landing_page_url": "https://x"},
    "keywords": [{"display_name": "veau"}],
}


def test_rebuild_abstract_from_inverted_index():
    assert rebuild_abstract(OPENALEX_WORK["abstract_inverted_index"]) == "Le colostrum protège le veau"


def test_article_from_openalex_and_license_filter():
    a = article_from_openalex(OPENALEX_WORK)
    assert a.id == "10.20870/productions-animales.2024.37.1.0000"
    assert a.authors == ["Marie Curie", "Louis Pasteur"] and a.year == 2024
    assert is_reusable(a)
    assert a.citation_label() == "Curie et al., 2024"
    assert not is_reusable(a.model_copy(update={"license": "cc-by-nc"}))


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def raise_for_status(self):
        pass

    def json(self):
        return self._data


class FakeSession:
    """Simule l'API OpenAlex : deux pages reliées par un curseur."""

    def __init__(self):
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append(dict(params))
        if params["cursor"] == "*":
            return FakeResponse({"results": [OPENALEX_WORK], "meta": {"next_cursor": "page2"}})
        return FakeResponse({"results": [dict(OPENALEX_WORK, id="https://openalex.org/W2")], "meta": {"next_cursor": None}})


def test_openalex_pagination_with_cursor():
    s = FakeSession()
    works = list(iter_openalex(max_items=10, session=s, delay_s=0))
    assert len(works) == 2 and s.calls[1]["cursor"] == "page2"
    assert "publication_year:>2019" in s.calls[0]["filter"]


def test_parse_oai_dublin_core():
    xml = """<?xml version="1.0"?><OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords><record>
    <header><identifier>oai:x:1</identifier></header><metadata>
    <oai_dc:dc xmlns:oai_dc="http://www.openarchives.org/OAI/2.0/oai_dc/" xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:title>Titre</dc:title><dc:creator>Dupont, Jean</dc:creator><dc:date>2023-05-01</dc:date>
    <dc:identifier>https://revue.org/article/view/42</dc:identifier><dc:identifier>10.1234/abc</dc:identifier>
    <dc:rights>https://creativecommons.org/licenses/by/4.0</dc:rights></oai_dc:dc></metadata></record>
    <resumptionToken>tok</resumptionToken></ListRecords></OAI-PMH>"""
    arts, token = parse_oai_records(xml)
    assert token == "tok" and arts[0].doi == "10.1234/abc" and arts[0].license == "cc-by"
    assert arts[0].pdf_url == "https://revue.org/article/download/42" and arts[0].year == 2023


def test_catalog_is_idempotent():
    cat = Catalog(Path(tempfile.mkdtemp()) / "c.sqlite")
    a = article_from_openalex(OPENALEX_WORK)
    cat.upsert(a)
    cat.upsert(a)
    assert cat.count() == 1 and cat.get(a.id).title == "Une synthèse"


def test_text_utils():
    assert tokenize("Les vaches laitières") == ["vache", "laitiere"]
    assert len(split_sentences("Phrase un. Phrase deux ! Et trois ?")) == 3


def test_chunking_strategies():
    text = " ".join(f"mot{i}" for i in range(500))
    parts = fixed_size(text, 200, 50)
    assert len(parts) == 3 and parts[1].split()[0] == "mot150"           # chevauchement de 50 mots
    sents = [f"Phrase numéro {i} sur les vaches." for i in range(40)]
    packed = sentence_pack(sents, 50)
    assert all(p.endswith(".") for p in packed)                         # jamais de phrase coupée
    sem = semantic_split(sents[:5] + [f"Le méthane du rumen numéro {i}." for i in range(5)], HashingEmbedder(128))
    assert len(sem) >= 2


def test_chunk_document_keeps_provenance():
    art = Article(id="a1", title="T", authors=["A B"], year=2021, doi="10.1/x", license="cc-by")
    doc = ParsedDocument(article_id="a1", sections=[Section(title="Intro", text="La vache produit du lait. " * 30, page=2)],
                         tables=["| a | b |\n|---|---|\n| 1 | 2 |"])
    chunks = chunk_document(doc, art, "structural", 60)
    assert chunks and chunks[0].meta["doi"] == "10.1/x" and chunks[0].page == 2 and chunks[0].section == "Intro"


def test_gazetteer_ner_longest_match_and_plurals():
    ents = GazetteerNER().extract("Le bien-être animal et les mammites des vaches Prim'Holstein.")
    summary = summarize_entities(ents)
    assert summary["enjeu"] == ["bien-être animal"]
    assert summary["pathologie"] == ["mammite"] and summary["espece"] == ["bovin"]
    assert summary["race"] == ["Prim'Holstein"]


def test_weak_label():
    assert weak_label("La ration et la digestibilité des fourrages") == "nutrition"
    assert weak_label("Rien à voir") is None


def test_pipeline_is_incremental():
    ctx = make_ctx()
    n = ctx.store.count()
    stats = ctx.index_samples(log=lambda *a: None)
    assert stats["skipped"] == 8 and ctx.store.count() == n                 # rien n'est ré-indexé
    manifest = json.loads(ctx.settings.path("index", "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["articles"]) == 8 and manifest["signature"]["embedder"].startswith("hashing")
    c = ctx.store.all_chunks()[0]
    assert {"entities", "species", "themes", "label"} <= set(c.meta)


def test_pdf_parsing_end_to_end():
    """Crée un vrai PDF (matplotlib) puis le lit avec pypdf : le parsing fonctionne sur un fichier réel."""
    require("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from transfia.ingestion.parse import PyPDFParser

    path = Path(tempfile.mkdtemp()) / "doc.pdf"
    fig = plt.figure(figsize=(8.27, 11.69))
    fig.text(0.1, 0.9, "Le colostrum protège le veau contre les diarrhées néonatales.", fontsize=11)
    fig.text(0.1, 0.85, "La prise doit être précoce, dans les premières heures de vie.", fontsize=11)
    fig.savefig(path)
    plt.close(fig)
    doc = PyPDFParser().parse(str(path), "pdf-test")
    assert "colostrum" in doc.full_text and doc.n_pages == 1


# ---------------------------------------- qualité des données (cas réel : clusters "bibliographie" et "br")
def test_reference_list_is_detected_but_prose_with_citations_is_not():
    from transfia.ingestion.chunking import looks_like_references

    refs = ("Richardson, C. M., Stothard, P., & Lassen, J. (2021). Breeding for reduced methane emission. "
            "Journal of Dairy Science, 104(8), 8983-9001. https://doi.org/10.3168/jds.2020-19889 "
            "Martin, C., Morgavi, D. P., & Doreau, M. (2010). Methane mitigation in ruminants. Animal, 4(3), "
            "351-365. https://doi.org/10.1017/S1751731109990620 Boichard, D., & Brochard, M. (2012). New phenotypes.")
    prose = ("Dans le contexte d'une crise du multilatéralisme (Lamy, 2021 ; Siroën, 2021 ; Mathonnière, 2023) et "
             "pour faire face à certains défis économiques et plus globaux, dont le changement climatique et la "
             "stabilité des marchés, l'Union européenne a renforcé ses accords commerciaux avec plusieurs pays "
             "tiers, ce qui modifie les flux de produits animaux et la place des filières françaises.")
    assert looks_like_references(refs) and not looks_like_references(prose)


def test_reference_section_and_table_br_are_cleaned():
    from transfia.core.models import Article, ParsedDocument, Section
    from transfia.ingestion.chunking import chunk_document

    text = "Le colostrum apporte au veau des anticorps indispensables dans les premières heures de vie. " * 3
    doc = ParsedDocument(article_id="a", sections=[Section(title="Introduction", text=text),
                                                  Section(title="Références bibliographiques", text=text)],
                         tables=["|Abréviation|Défi<br>nition|\n|---|---|\n|CI|Capacité d'ingestion du veau en élevage|"])
    stats = {}
    chunks = chunk_document(doc, Article(id="a", title="t"), stats=stats)
    assert stats["references_skipped"] == 1 and all(c.section != "Références bibliographiques" for c in chunks)
    assert all("<br>" not in c.text for c in chunks)


def test_reindexing_removes_stale_chunks():
    import numpy as np

    from transfia.core.models import Chunk
    from transfia.retrieval.vector_store import InMemoryVectorStore

    st = InMemoryVectorStore()
    st.upsert([Chunk(id=f"a#{i}", article_id="a", text="x") for i in range(3)] + [Chunk(id="b#0", article_id="b", text="y")],
              np.eye(4, dtype=np.float32))
    assert st.delete_article("a") == 3 and [c.id for c in st.all_chunks()] == ["b#0"] and st.vectors.shape == (1, 4)
