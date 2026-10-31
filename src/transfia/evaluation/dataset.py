"""Jeu de test (golden dataset) pour le banc d'éval (F11).

Trois sources d'items, à garder SÉPARÉES (champ ``source``) :
1. "human"     : questions écrites (ou validées) par des experts métier. La référence.
2. "synthetic" : questions générées par un LLM à partir d'un passage -> réponse vérifiable dans le texte.
3. "pseudo"    : SANS LLM : une phrase du passage, dont on retire des mots, sert de requête.
                 Une baseline gratuite pour évaluer la RECHERCHE (pas la génération).

Piège : ne pas régler le système sur le jeu de test (sur-apprentissage du jeu de test).
-> on garde une partie "dev" pour régler, et une partie "test" pour mesurer.
"""
from __future__ import annotations

import re

import json
import random
from pathlib import Path

from pydantic import BaseModel, Field

from transfia.core.models import Chunk
from transfia.core.text import split_sentences, tokenize
from transfia.llm.base import Message, StructuredOutputError, chat_structured


class EvalItem(BaseModel):
    id: str
    question: str
    reference_answer: str = ""
    source_article_ids: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    source: str = "human"
    tags: list[str] = Field(default_factory=list)
    split: str = "test"


def load_dataset(path: Path) -> list[EvalItem]:
    return [EvalItem(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_dataset(items: list[EvalItem], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(it.model_dump_json() + "\n")


def pseudo_queries(chunks: list[Chunk], n: int = 50, drop: float = 0.4, seed: int = 0) -> list[EvalItem]:
    rng = random.Random(seed)
    pool = [c for c in chunks if len(c.text.split()) > 30]
    rng.shuffle(pool)
    items = []
    for c in pool[:n]:
        sents = [s for s in split_sentences(c.text) if len(s.split()) >= 8]
        if not sents:
            continue
        words = rng.choice(sents).split()
        kept = [w for w in words if rng.random() > drop] or words[:5]
        items.append(EvalItem(id=f"pseudo-{c.id}", question=" ".join(kept), source_article_ids=[c.article_id],
                              source_chunk_ids=[c.id], source="pseudo"))
    return items


class _QA(BaseModel):
    question: str
    reponse: str


_NOT_SELF_CONTAINED = re.compile(
    r"(?:\b(?:ce|cet|cette|ces)\s+|\bdans\s+(?:le|la|les|l')\s*)(?:passage|texte|article|extrait|document|"
    r"étude|recherche|paragraphe|tableau|figure)s?\b", re.I)


def is_self_contained(question: str) -> bool:
    """Une question d'éval doit se comprendre SANS le passage sous les yeux (vu : « Quel est l'objectif
    principal de la recherche présentée dans ce passage ? », malgré la consigne qui l'interdisait)."""
    return not _NOT_SELF_CONTAINED.search(question)


def synthetic_questions(llm, chunks: list[Chunk], n: int = 20, seed: int = 0) -> list[EvalItem]:
    """Une question par passage. Filtre : la question ne doit pas recopier le passage, la réponse
    doit en reprendre les mots (sinon elle n'est pas vérifiable dans le texte)."""
    rng = random.Random(seed)
    pool = [c for c in chunks if len(c.text.split()) > 40]
    rng.shuffle(pool)
    items = []
    for c in pool:
        if len(items) >= n:
            break
        prompt = ("À partir du passage, écris UNE question qu'un formateur en élevage pourrait poser, dont la réponse "
                  "se trouve dans le passage, et la réponse courte. N'utilise pas les mots « le passage » ou « le texte ».\n\n"
                  f"Passage : {c.text}")
        try:
            qa, _ = chat_structured(llm, [Message("user", prompt)], _QA, retries=1, max_tokens=250)
        except StructuredOutputError:
            continue
        q_terms, p_terms, a_terms = set(tokenize(qa.question)), set(tokenize(c.text)), set(tokenize(qa.reponse))
        copied = len(q_terms & p_terms) / max(len(q_terms), 1) > 0.95 and len(q_terms) > 8
        grounded = len(a_terms & p_terms) / max(len(a_terms), 1) >= 0.5
        if not copied and grounded and is_self_contained(qa.question):
            items.append(EvalItem(id=f"syn-{c.id}", question=qa.question, reference_answer=qa.reponse,
                                  source_article_ids=[c.article_id], source_chunk_ids=[c.id], source="synthetic"))
    return items


def split_dev_test(items: list[EvalItem], dev_ratio: float = 0.3, seed: int = 0) -> list[EvalItem]:
    rng = random.Random(seed)
    for it in items:
        it.split = "dev" if rng.random() < dev_ratio else "test"
    return items


def dataset_coverage(items: list[EvalItem], indexed_article_ids: set[str]) -> float:
    """Part des questions dont AU MOINS UNE source attendue est dans l'index.
    Garde-fou : un jeu d'éval écrit pour un autre corpus donne 0 partout (vu en vrai : golden_sample
    écrit pour les articles d'exemple, évalué sur les 10 vrais articles INRAE)."""
    with_src = [it for it in items if it.source_article_ids]
    if not with_src:
        return 1.0
    return sum(any(a in indexed_article_ids for a in it.source_article_ids) for it in with_src) / len(with_src)


# Questions du DOMAINE (élevage) mais ABSENTES du corpus : le bon comportement est de REFUSER.
# Sans elles, refusal_correct = 1.0 ne prouve rien (vu en vrai : 5 questions, toutes avec réponse).
# Choisies hors des 10 articles INRAE 2026 (vérifié pour poules pondeuses et grippe aviaire : refus constatés).
OUT_OF_CORPUS = [
    "Comment lutter contre le varroa dans les ruches ?",
    "Quelle densité d'élevage pour les huîtres en parc ?",
    "Comment nourrir des escargots en héliciculture ?",
    "À quelle saison tondre les alpagas ?",
    "Comment améliorer le bien-être des poules pondeuses en cage ?",
    "Quel est le meilleur traitement contre la grippe aviaire ?",
    "Quelle ration donner à une autruche en croissance ?",
    "Comment prévenir la maladie aléoutienne chez le vison ?",
]


def out_of_corpus_items(n: int = 8) -> list[EvalItem]:
    return [EvalItem(id=f"hors-{i:02d}", question=q, source="hors_corpus", tags=["hors_corpus"])
            for i, q in enumerate(OUT_OF_CORPUS[:n])]
