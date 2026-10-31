"""Parsing des documents (F2) : PDF -> document STRUCTURÉ (sections, tableaux, figures).

Trois parseurs interchangeables (même interface ``parse(path, article_id)``) :
1. ``PyMuPDFParser`` (défaut, léger) : lit le texte avec sa TAILLE DE POLICE. On détecte les
   titres "from scratch" : ligne courte écrite plus gros que le corps du texte. Il extrait aussi
   les tableaux (``find_tables``) et les images.
2. ``PyPDFParser`` (ultra-léger) : texte brut page par page, sans structure.
3. ``DoclingParser`` (le plus complet) : modèles de LAYOUT ANALYSIS (détection des zones : titre,
   paragraphe, tableau, figure) et de TABLE STRUCTURE RECOGNITION. Plus lourd (8 Go = juste).

OCR (pages scannées) : Tesseract via pytesseract, seulement si une page n'a pas de texte.
VLM (figures) : description générée par un petit modèle vision-langage, MARQUÉE comme générée
(un VLM peut inventer des valeurs lues sur un graphique !).
"""
from __future__ import annotations

import base64
import re
import statistics
from pathlib import Path

from transfia.core.models import Figure, ParsedDocument, Section
from transfia.observability.tracing import span

_NOISE = re.compile(r"^(\d+|page \d+|inrae productions animales.*|https?://\S+)$", re.I)


def _clean(text: str) -> str:
    text = re.sub(r"-\n(?=[a-zà-ÿ])", "", text)          # césures de fin de ligne
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


class PyMuPDFParser:
    name = "pymupdf"

    def __init__(self, ocr: bool = False, extract_images: bool = False):
        try:
            import pymupdf as fitz            # nom actuel du paquet
        except ImportError:
            try:
                import fitz                   # ancien nom
            except ImportError as exc:
                raise ImportError('pip install -e ".[local]" (pymupdf)') from exc
        self.fitz = fitz
        self.ocr = ocr
        self.extract_images = extract_images

    def parse(self, path: str, article_id: str) -> ParsedDocument:
        with span("parse.pymupdf", path=Path(path).name) as s:
            doc = self.fitz.open(path)
            lines: list[tuple[float, str, int]] = []     # (taille de police, texte, page)
            tables: list[str] = []
            figures: list[Figure] = []
            for pno, page in enumerate(doc, start=1):
                d = page.get_text("dict")
                if not d.get("blocks") and self.ocr:
                    lines.append((10.0, self._ocr_page(page), pno))
                for block in d.get("blocks", []):
                    if block.get("type") == 1 and self.extract_images:
                        figures.append(Figure(page=pno, caption=""))
                    for line in block.get("lines", []):
                        spans = line.get("spans", [])
                        txt = "".join(sp["text"] for sp in spans).strip()
                        if txt and not _NOISE.match(txt):
                            size = max(sp["size"] for sp in spans)
                            lines.append((size, txt, pno))
                try:
                    for t in page.find_tables().tables:
                        tables.append(t.to_markdown())
                except Exception:
                    pass
            sections = self._sections_from_lines(lines)
            s.set(n_pages=len(doc), n_sections=len(sections), n_tables=len(tables))
            return ParsedDocument(article_id=article_id, sections=sections, tables=tables, figures=figures,
                                  n_pages=len(doc), parser=self.name)

    @staticmethod
    def _sections_from_lines(lines: list[tuple[float, str, int]]) -> list[Section]:
        """Détection de titres from scratch : police > 1,15 × médiane du corps ET ligne courte."""
        if not lines:
            return []
        body = statistics.median(size for size, _, _ in lines)
        sections: list[Section] = []
        title, buf, page = "Début", [], lines[0][2]
        for size, txt, pno in lines:
            is_heading = size >= body * 1.15 and len(txt) < 120 and not txt.endswith(".")
            if is_heading:
                if buf:
                    sections.append(Section(title=title, text=_clean("\n".join(buf)), page=page))
                title, buf, page = txt, [], pno
            else:
                buf.append(txt)
        if buf:
            sections.append(Section(title=title, text=_clean("\n".join(buf)), page=page))
        return [s for s in sections if len(s.text) > 40]

    def _ocr_page(self, page) -> str:
        try:
            import io

            import pytesseract
            from PIL import Image

            pix = page.get_pixmap(dpi=200)
            return pytesseract.image_to_string(Image.open(io.BytesIO(pix.tobytes("png"))), lang="fra")
        except Exception:
            return ""

    def page_images(self, path: str, max_images: int = 5) -> list[tuple[int, bytes]]:
        """Extrait les images (pour la description par VLM)."""
        doc = self.fitz.open(path)
        out = []
        for pno, page in enumerate(doc, start=1):
            for img in page.get_images(full=True):
                pix = self.fitz.Pixmap(doc, img[0])
                if pix.width < 150 or pix.height < 150:
                    continue
                if pix.n > 4:
                    pix = self.fitz.Pixmap(self.fitz.csRGB, pix)
                out.append((pno, pix.tobytes("png")))
                if len(out) >= max_images:
                    return out
        return out


class PyPDFParser:
    name = "pypdf"

    def parse(self, path: str, article_id: str) -> ParsedDocument:
        from pypdf import PdfReader

        reader = PdfReader(path)
        sections = []
        for i, page in enumerate(reader.pages, start=1):
            txt = _clean(page.extract_text() or "")
            if len(txt) > 40:
                sections.append(Section(title=f"Page {i}", text=txt, page=i))
        return ParsedDocument(article_id=article_id, sections=sections, n_pages=len(reader.pages), parser=self.name)


class DoclingParser:
    name = "docling"

    def __init__(self):
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise ImportError('pip install -e ".[docling]"') from exc
        self.converter = DocumentConverter()

    def parse(self, path: str, article_id: str) -> ParsedDocument:
        with span("parse.docling", path=Path(path).name):
            res = self.converter.convert(path)
            md = res.document.export_to_markdown()
        sections, title, buf = [], "Début", []
        tables = []
        for line in md.splitlines():
            if line.startswith("#"):
                if buf:
                    sections.append(Section(title=title, text=_clean("\n".join(buf))))
                title, buf = line.lstrip("# ").strip(), []
            elif line.startswith("|"):
                tables.append(line)
                buf.append(line)
            else:
                buf.append(line)
        if buf:
            sections.append(Section(title=title, text=_clean("\n".join(buf))))
        return ParsedDocument(article_id=article_id, sections=[s for s in sections if len(s.text) > 40],
                              tables=["\n".join(tables)] if tables else [], parser=self.name)


class TextParser:
    """Pour les résumés seuls ou les documents déjà en texte (corpus d'exemple, tests)."""

    name = "text"

    def parse_sections(self, article_id: str, sections: list[dict]) -> ParsedDocument:
        return ParsedDocument(article_id=article_id, parser=self.name,
                              sections=[Section(title=s.get("title", ""), text=s["text"], page=s.get("page"))
                                        for s in sections])

    def parse(self, path: str, article_id: str) -> ParsedDocument:
        txt = Path(path).read_text(encoding="utf-8")
        return ParsedDocument(article_id=article_id, sections=[Section(title="Texte", text=txt)], parser=self.name)


def build_parser(settings):
    kind = settings.ingestion.parser
    if kind == "docling":
        return DoclingParser()
    if kind == "pypdf":
        return PyPDFParser()
    if kind == "text":
        return TextParser()
    return PyMuPDFParser(ocr=settings.ingestion.ocr, extract_images=settings.ingestion.describe_figures)


def describe_figure(llm, png_bytes: bytes, model: str | None = None) -> str:
    """VLM : l'image est encodée en tokens visuels et injectée dans le modèle de langage."""
    from transfia.llm.base import Message

    msg = Message("user", "Décris cette figure d'article scientifique en 3 phrases, en français. "
                          "Si tu lis des valeurs, précise qu'elles sont approximatives.",
                  images=[base64.b64encode(png_bytes).decode()])
    return llm.chat([msg], model=model, temperature=0.0).content.strip()
