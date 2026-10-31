"""Chunking (F3) : découper les documents en morceaux adaptés à la recherche.

Trois stratégies, à COMPARER dans le banc d'éval :
1. TAILLE FIXE : N mots avec chevauchement (overlap) pour ne pas couper une idée en deux.
2. STRUCTUREL : on suit les sections du document ; on redécoupe seulement les sections trop longues.
3. SÉMANTIQUE : on coupe là où la similarité entre phrases consécutives CHUTE (changement de sujet).

Compromis : trop petit = perte de contexte ; trop gros = bruit dans le prompt et coût en tokens.
Chaque chunk garde ses MÉTADONNÉES (article, section, page, auteurs, année, DOI) : c'est ce qui
permet de citer les sources.
"""
from __future__ import annotations

import re

import numpy as np

from transfia.core.models import Article, Chunk, ParsedDocument
from transfia.core.text import split_sentences


def fixed_size(text: str, size: int = 220, overlap: int = 40) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap doit être < size")
    words = text.split()
    if not words:
        return []
    out, step = [], size - overlap
    for start in range(0, len(words), step):
        out.append(" ".join(words[start:start + size]))
        if start + size >= len(words):
            break
    return out


def sentence_pack(sentences: list[str], max_words: int = 220, overlap_sentences: int = 1) -> list[str]:
    """Regroupe des phrases entières jusqu'à max_words (ne coupe jamais une phrase)."""
    out, buf, n = [], [], 0
    for s in sentences:
        w = len(s.split())
        if buf and n + w > max_words:
            out.append(" ".join(buf))
            buf = buf[-overlap_sentences:] if overlap_sentences else []
            n = sum(len(x.split()) for x in buf)
        buf.append(s)
        n += w
    if buf:
        out.append(" ".join(buf))
    return out


def semantic_split(sentences: list[str], embedder, percentile: float = 20, max_words: int = 300) -> list[str]:
    """Coupe aux plus fortes baisses de similarité entre phrases voisines.

    Seuil = le p-ième percentile des similarités consécutives (les 20 % les plus faibles = ruptures).
    """
    if len(sentences) <= 2:
        return [" ".join(sentences)] if sentences else []
    vecs = embedder.embed_documents(sentences)
    sims = np.sum(vecs[:-1] * vecs[1:], axis=1)
    threshold = np.percentile(sims, percentile)
    out, buf = [], [sentences[0]]
    for s, sim in zip(sentences[1:], sims):
        too_long = len(" ".join(buf).split()) > max_words
        if sim < threshold or too_long:
            out.append(" ".join(buf))
            buf = []
        buf.append(s)
    out.append(" ".join(buf))
    return out


# --- qualité des données (cas vécu : clustering) ----------------------------------------------
# Un cluster entier était fait de BIBLIOGRAPHIE ("of, and, https, org, doi") et un autre portait "br"
# (balises <br> des tableaux). Ces passages polluent aussi la recherche (vu : une référence anglaise
# remontée en 3e résultat sur "colostrum"). On les écarte À L'INGESTION : c'est là qu'on les corrige le mieux.
REF_SECTION = re.compile(r"^\s*(?:\d+[.)]?\s*)?(?:r[ée]f[ée]rences?(?: bibliographiques)?|bibliograph(?:ie|y)|"
                         r"references|literature cited)\b", re.I)
_YEAR = re.compile(r"\(?\b(?:19|20)\d{2}[a-z]?\)?[.,;:]")
_LINK = re.compile(r"\bdoi\b|https?://|\bdoi\.org", re.I)
_INITIALS = re.compile(r"\b[A-Z]\.(?:\s?[A-Z]\.)*\s?(?:,|&|\()")


def looks_like_references(text: str, threshold: float = 8.0) -> bool:
    """Densité de "marqueurs bibliographiques" (années, DOI/URL, initiales) pour 100 mots.
    Prose normale avec citations "(Lamy, 2021 ; Siroën, 2021)" ≈ 2 ; liste de références ≈ 20 à 40."""
    words = max(len(text.split()), 1)
    score = len(_YEAR.findall(text)) + 2 * len(_LINK.findall(text)) + len(_INITIALS.findall(text))
    return score * 100 / words >= threshold


def clean_table(table: str) -> str:
    return re.sub(r"<br\s*/?>", " ", table, flags=re.I)


def chunk_document(doc: ParsedDocument, article: Article, strategy: str = "structural", chunk_words: int = 220,
                   overlap_words: int = 40, embedder=None, stats: dict | None = None) -> list[Chunk]:
    """``stats`` (optionnel) reçoit {"references_skipped": n} pour le journal d'ingestion."""
    meta_base = {
        "title": article.title, "authors": article.authors[:6], "year": article.year, "doi": article.doi,
        "label": article.citation_label(), "license": article.license,
    }
    pieces: list[tuple[str, str, int | None]] = []          # (texte, section, page)
    refs_skipped = 0
    if strategy == "fixed":
        for t in fixed_size(doc.full_text, chunk_words, overlap_words):
            pieces.append((t, "", None))
    else:
        for sec in doc.sections:
            if REF_SECTION.match(sec.title or ""):     # toute la section "Références" : écartée
                refs_skipped += len(sentence_pack(split_sentences(sec.text), chunk_words)) or 1   # en PASSAGES
                continue
            sents = split_sentences(sec.text)
            if strategy == "semantic" and embedder is not None:
                parts = semantic_split(sents, embedder, max_words=chunk_words + 80)
            else:
                parts = sentence_pack(sents, chunk_words)
            for t in parts:
                pieces.append((t, sec.title, sec.page))
    for i, table in enumerate(doc.tables):
        pieces.append((f"Tableau {i + 1} :\n{clean_table(table)}", "Tableaux", None))
    for fig in doc.figures:
        if fig.description:
            tag = " (description générée par IA)" if fig.generated else ""
            pieces.append((f"Figure p.{fig.page}{tag} : {fig.caption} {fig.description}", "Figures", fig.page))
    chunks = []
    for i, (text, section, page) in enumerate(pieces):
        if len(text.split()) < 8:
            continue
        if section != "Tableaux" and looks_like_references(text):   # bibliographie hors section titrée
            refs_skipped += 1
            continue
        chunks.append(Chunk(id=f"{article.id}#{i:03d}", article_id=article.id, text=text, section=section,
                            page=page, meta={**meta_base, "section": section}))
    if stats is not None:
        stats["references_skipped"] = refs_skipped
    return chunks
