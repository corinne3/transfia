"""Collecte du corpus (F1) : catalogue des articles + téléchargement des PDF.

Source principale : l'API OpenAlex (catalogue ouvert de la littérature scientifique).
- Revue INRAE Productions Animales : source OpenAlex S4210213780.
- Filtres : année >= 2020 ET licence CC BY (vérifiée ARTICLE PAR ARTICLE, pas seulement au
  niveau de la revue).
- Pagination par curseur (cursor=*), délai entre requêtes, User-Agent explicite.

Alternative fournie : un moissonneur OAI-PMH générique (protocole standard des revues OJS).
L'adresse exacte de l'entrepôt OAI de la revue reste à vérifier avant usage.

IDEMPOTENCE : relancer la collecte ne retélécharge que ce qui manque.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterator

import requests

from transfia.core.models import Article

OPENALEX = "https://api.openalex.org/works"
PRODUCTIONS_ANIMALES = "S4210213780"
ALLOWED_LICENSES = {"cc-by", "cc0", "public-domain"}


def rebuild_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """OpenAlex stocke les résumés en "index inversé" {mot: [positions]} : on le remet à plat."""
    if not inverted_index:
        return ""
    positions = [(pos, word) for word, poss in inverted_index.items() for pos in poss]
    return " ".join(word for _, word in sorted(positions))


def article_from_openalex(work: dict) -> Article:
    oa = work.get("best_oa_location") or work.get("primary_location") or {}
    doi = (work.get("doi") or "").replace("https://doi.org/", "") or None
    license_ = (oa.get("license") or (work.get("primary_location") or {}).get("license") or "").lower() or None
    return Article(
        id=doi or work["id"].rsplit("/", 1)[-1],
        title=(work.get("title") or work.get("display_name") or "").strip(),
        authors=[a["author"]["display_name"] for a in work.get("authorships", []) if a.get("author")],
        year=work.get("publication_year"),
        doi=doi,
        abstract=rebuild_abstract(work.get("abstract_inverted_index")),
        keywords=[k.get("display_name", "") for k in work.get("keywords", [])][:10],
        license=license_,
        pdf_url=oa.get("pdf_url"),
        landing_url=oa.get("landing_page_url") or work.get("doi"),
        source="openalex",
    )


def iter_openalex(source_id: str = PRODUCTIONS_ANIMALES, year_min: int = 2020, max_items: int = 100,
                  email: str | None = None, session=None, delay_s: float = 0.2) -> Iterator[dict]:
    session = session or requests.Session()
    params = {
        "filter": f"primary_location.source.id:{source_id},publication_year:>{year_min - 1},is_oa:true",
        "per-page": min(max_items, 50),
        "cursor": "*",
        "sort": "publication_date:desc",
    }
    if email:
        params["mailto"] = email          # "polite pool" d'OpenAlex
    headers = {"User-Agent": f"TransfIA/0.1 (open source; {email or 'contact non renseigné'})"}
    seen = 0
    while seen < max_items:
        r = session.get(OPENALEX, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json()
        for work in data.get("results", []):
            yield work
            seen += 1
            if seen >= max_items:
                return
        cursor = data.get("meta", {}).get("next_cursor")
        if not cursor or not data.get("results"):
            return
        params["cursor"] = cursor
        time.sleep(delay_s)


def is_reusable(article: Article) -> bool:
    return (article.license or "") in ALLOWED_LICENSES


# ---------------------------------------------------------------------------
# Catalogue SQLite
# ---------------------------------------------------------------------------
class Catalog:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(path) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS articles (
                id TEXT PRIMARY KEY, data TEXT NOT NULL, pdf_path TEXT, status TEXT DEFAULT 'new')""")

    def upsert(self, a: Article) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO articles (id, data) VALUES (?, ?) "
                       "ON CONFLICT(id) DO UPDATE SET data=excluded.data", (a.id, a.model_dump_json()))

    def set_pdf(self, article_id: str, pdf_path: str | None, status: str) -> None:
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE articles SET pdf_path=?, status=? WHERE id=?", (pdf_path, status, article_id))

    def all(self) -> list[tuple[Article, str | None, str]]:
        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT data, pdf_path, status FROM articles ORDER BY id").fetchall()
        return [(Article.model_validate_json(d), p, s) for d, p, s in rows]

    def get(self, article_id: str) -> Article | None:
        with sqlite3.connect(self.path) as db:
            row = db.execute("SELECT data FROM articles WHERE id=?", (article_id,)).fetchone()
        return Article.model_validate_json(row[0]) if row else None

    def count(self) -> int:
        with sqlite3.connect(self.path) as db:
            return db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]


def safe_filename(article_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", article_id)[:120]


def download_pdf(url: str, dest: Path, session=None, email: str | None = None) -> bool:
    if dest.exists() and dest.stat().st_size > 1000:
        return True                                    # idempotence
    session = session or requests.Session()
    headers = {"User-Agent": f"TransfIA/0.1 (open source; {email or 'n/a'})"}
    r = session.get(url, headers=headers, timeout=60)
    if r.status_code != 200 or not r.content.startswith(b"%PDF"):
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return True


def collect(settings, max_items: int = 30, year_min: int = 2020, download: bool = True,
            session=None, log=print) -> dict:
    catalog = Catalog(settings.path("catalog.sqlite"))
    raw_dir = settings.path("raw", ".keep").parent
    stats = {"seen": 0, "kept": 0, "rejected_license": 0, "pdf_ok": 0, "pdf_missing": 0}
    for work in iter_openalex(year_min=year_min, max_items=max_items, email=settings.contact_email, session=session):
        stats["seen"] += 1
        art = article_from_openalex(work)
        if not is_reusable(art):
            stats["rejected_license"] += 1
            continue
        catalog.upsert(art)
        stats["kept"] += 1
        if download and art.pdf_url:
            dest = raw_dir / f"{safe_filename(art.id)}.pdf"
            try:
                ok = download_pdf(art.pdf_url, dest, session=session, email=settings.contact_email)
            except requests.RequestException:
                ok = False
            catalog.set_pdf(art.id, str(dest) if ok else None, "pdf" if ok else "abstract_only")
            stats["pdf_ok" if ok else "pdf_missing"] += 1
            time.sleep(0.5)                            # politesse envers le serveur
        else:
            catalog.set_pdf(art.id, None, "abstract_only")
            stats["pdf_missing"] += 1
        log(f"  ✓ {art.year} {art.title[:70]}")
    return stats


# ---------------------------------------------------------------------------
# Alternative : OAI-PMH (Dublin Core)
# ---------------------------------------------------------------------------
NS = {"oai": "http://www.openarchives.org/OAI/2.0/", "dc": "http://purl.org/dc/elements/1.1/",
      "oai_dc": "http://www.openarchives.org/OAI/2.0/oai_dc/"}


def parse_oai_records(xml_text: str) -> tuple[list[Article], str | None]:
    root = ET.fromstring(xml_text)
    articles = []
    for rec in root.iterfind(".//oai:record", NS):
        md = rec.find(".//oai_dc:dc", NS)
        if md is None:
            continue

        def all_(tag: str) -> list[str]:
            return [e.text.strip() for e in md.findall(f"dc:{tag}", NS) if e.text]

        idents = all_("identifier")
        doi = next((i.split("doi.org/")[-1] for i in idents if "10." in i), None)
        url = next((i for i in idents if i.startswith("http") and "/article/view/" in i), None)
        rights = " ".join(all_("rights")).lower()
        lic = "cc-by" if "by/4.0" in rights or "cc by" in rights or "cc-by" in rights else (rights or None)
        date = (all_("date") or [""])[0]
        articles.append(Article(
            id=doi or rec.findtext(".//oai:identifier", default="", namespaces=NS),
            title=(all_("title") or [""])[0], authors=all_("creator"),
            year=int(date[:4]) if date[:4].isdigit() else None, doi=doi,
            abstract=(all_("description") or [""])[0], keywords=all_("subject"), license=lic,
            pdf_url=url.replace("/article/view/", "/article/download/") if url else None,
            landing_url=url, source="oai-pmh"))
    token = root.find(".//oai:resumptionToken", NS)
    return articles, (token.text if token is not None and token.text else None)


def iter_oai(base_url: str, session=None) -> Iterator[Article]:
    session = session or requests.Session()
    params = {"verb": "ListRecords", "metadataPrefix": "oai_dc"}
    while True:
        r = session.get(base_url, params=params, timeout=60)
        r.raise_for_status()
        arts, token = parse_oai_records(r.text)
        yield from arts
        if not token:
            return
        params = {"verb": "ListRecords", "resumptionToken": token}
        time.sleep(1)


def load_sample_articles(path: Path) -> list[tuple[Article, list[dict]]]:
    """Corpus d'exemple (data/samples) : articles synthétiques avec sections, pour tester hors ligne."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            d = json.loads(line)
            sections = d.pop("sections", [])
            out.append((Article(**d), sections))
    return out
