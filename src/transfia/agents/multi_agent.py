"""Atelier multi-agents (F9) : rédacteur -> relecteur scientifique -> vulgarisateur.

Codé comme une petite MACHINE À ÉTATS (from scratch), exactement ce que LangGraph formalise :
- un ÉTAT partagé (brouillon, sources, problèmes, nombre de tours) circule entre des NŒUDS ;
- une ARÊTE CONDITIONNELLE choisit la suite : problèmes et tours < max -> retour au rédacteur,
  sinon -> vulgarisateur.
Pattern "REFLECTION / CRITIQUE" : un agent produit, un autre critique, le premier corrige.

Le relecteur utilise de préférence un modèle d'une AUTRE FAMILLE que le rédacteur, pour
limiter les angles morts partagés (section 11.3).

Leçons d'un essai réel (qwen2.5:1.5b rédacteur, llama3.2:1b relecteur, « le colostrum », grand public) :
1. BOUCLE À VIDE : 3 tours avec « 0 problème » : le relecteur renvoyait valide=false SANS problème (ou une
   sortie illisible) -> l'arête renvoyait au rédacteur pour rien. On route sur les PROBLÈMES, pas sur le drapeau.
2. RELECTEUR COMPLAISANT : un petit juge valide tout. On lui ajoute un relecteur AUTOMATIQUE (règles :
   vérification lexicale de chaque phrase contre les passages) : juge LLM + règles.
3. DERNIÈRE ÉTAPE NON VÉRIFIÉE : le vulgarisateur a écrit « Les vaches Charolaise ont des colostrums faibles »
   alors que la source dit « 20 % des colostrums ». On contrôle aussi le texte FINAL : chiffres perdus
   (perte de nuance) et phrases sans appui dans les passages.
"""
from __future__ import annotations

import re

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field

from transfia.core.models import Hit
from transfia.core.prompts import render
from transfia.core.text import tokenize, truncate_words
from transfia.llm.base import Message, StructuredOutputError, chat_structured
from transfia.observability.tracing import span


class Probleme(BaseModel):
    affirmation: str
    statut: Literal["soutenue", "non_soutenue", "contredite"]
    explication: str = ""


class Revue(BaseModel):
    problemes: list[Probleme] = Field(default_factory=list)
    valide: bool


@dataclass
class EditorialState:
    topic: str
    niveau: str
    passages: list[Hit]
    draft: str = ""
    reviews: list[Revue] = field(default_factory=list)
    rounds: int = 0
    final: str = ""
    history: list[str] = field(default_factory=list)      # trace lisible des nœuds visités
    final_warnings: list[str] = field(default_factory=list)

    def passages_text(self) -> str:
        return "\n".join(f"[{i + 1}] ({h.chunk.meta.get('label', '')}) {truncate_words(h.chunk.text, 200)}"
                         for i, h in enumerate(self.passages))

    @property
    def open_problems(self) -> list[Probleme]:
        return [p for p in (self.reviews[-1].problemes if self.reviews else []) if p.statut != "soutenue"]


class EditorialWorkflow:
    def __init__(self, writer_llm, reviewer_llm=None, simplifier_llm=None, skills=None, max_rounds: int = 3,
                 reviewer_model: str | None = None):
        self.writer = writer_llm
        self.reviewer = reviewer_llm or writer_llm
        self.simplifier = simplifier_llm or writer_llm
        self.skills = skills
        self.max_rounds = max_rounds
        self.reviewer_model = reviewer_model

    def _skill(self, name: str) -> str:
        return self.skills.load(name) if self.skills and name in self.skills else ""

    # ------------------------------------------------------------------ nœuds
    def node_write(self, st: EditorialState) -> EditorialState:
        with span("editorial.write", round=st.rounds):
            feedback = "\n".join(f"- « {p.affirmation} » : {p.statut} ({p.explication})" for p in st.open_problems)
            prompt = render("writer", topic=st.topic, passages=st.passages_text(), draft=st.draft, feedback=feedback)
            st.draft = self.writer.chat([Message("user", prompt)], max_tokens=900).content.strip()
            st.rounds += 1
            st.history.append(f"rédacteur (tour {st.rounds})")
            return st

    def node_review(self, st: EditorialState) -> EditorialState:
        with span("editorial.review", round=st.rounds) as s:
            prompt = render("reviewer", skill=self._skill("relecture-scientifique"), passages=st.passages_text(),
                            draft=st.draft)
            try:
                revue, _ = chat_structured(self.reviewer, [Message("user", prompt)], Revue, retries=1,
                                           max_tokens=900, model=self.reviewer_model)
            except StructuredOutputError:
                revue = None
                if self.reviewer is not self.writer:
                    # Vu en vrai : llama3.2:1b ne produit JAMAIS le JSON demandé (3 tours sur 3). Un relecteur
                    # d'une autre famille ne sert que s'il sait répondre : sinon, repli sur le modèle du rédacteur.
                    try:
                        revue, _ = chat_structured(self.writer, [Message("user", prompt)], Revue, retries=1,
                                                   max_tokens=900)
                        st.history.append("relecteur LLM illisible -> repli sur le modèle du rédacteur")
                    except StructuredOutputError:
                        revue = None
                if revue is None:
                    revue = Revue(problemes=[], valide=False)
                    st.history.append("relecteur LLM (sortie illisible)")
            n_llm = len([p for p in revue.problemes if p.statut != "soutenue"])
            # RELECTEUR AUTOMATIQUE (règles) : chaque phrase du brouillon est-elle soutenue par un passage ?
            auto = self.auto_review(st)
            known = {p.affirmation for p in revue.problemes}
            revue.problemes += [p for p in auto if p.affirmation not in known]
            # cohérence : "valide" = AUCUN problème ouvert (on ne se fie pas au drapeau du LLM)
            revue.valide = not any(p.statut != "soutenue" for p in revue.problemes)
            st.reviews.append(revue)
            st.history.append(f"relecteur ({n_llm} problème(s) LLM + {len(auto)} automatique(s))")
            s.set(valide=revue.valide)
            return st

    @staticmethod
    def auto_review(st: EditorialState) -> list[Probleme]:
        from transfia.rag.verify import attribute_citations, verify_answer

        passages = {i + 1: h.chunk.text for i, h in enumerate(st.passages)}
        text, _ = attribute_citations(st.draft, passages)
        return [Probleme(affirmation=c.sentence, statut="non_soutenue",
                         explication="vérification automatique : aucun passage cité ne soutient cette phrase")
                for c in verify_answer(text, passages) if not c.supported]

    def node_simplify(self, st: EditorialState) -> EditorialState:
        with span("editorial.simplify", niveau=st.niveau):
            prompt = render("simplifier", skill=self._skill("vulgarisation"), niveau=st.niveau, draft=st.draft)
            st.final = self.simplifier.chat([Message("user", prompt)], max_tokens=900).content.strip()
            st.history.append("vulgarisateur")
            return st

    # ------------------------------------------------------- arête conditionnelle
    def route_after_review(self, st: EditorialState) -> str:
        if not st.open_problems:                 # rien à corriger -> on avance (pas de tour à vide)
            return "simplify"
        return "write" if st.rounds < self.max_rounds else "prune"

    def node_prune(self, st: EditorialState) -> EditorialState:
        """Tours épuisés et problèmes persistants (vu : la même phrase signalée 3 fois, jamais corrigée) :
        le CODE retire les phrases non soutenues, plutôt que de les laisser passer à la vulgarisation."""
        bad = {p.affirmation.strip() for p in st.open_problems}
        from transfia.rag.verify import split_with_citations

        kept, removed = [], 0
        for sent in re.split(r"(?<=[.!?])\s+", st.draft):
            clean = split_with_citations(sent)
            if clean and clean[0][0].strip() in bad:
                removed += 1
                continue
            kept.append(sent)
        if removed and kept:
            st.draft = " ".join(kept)
            st.final_warnings.append(f"{removed} phrase(s) non soutenue(s) après {st.rounds} tours : RETIRÉE(S) du texte")
        st.history.append(f"élagage ({removed} phrase(s) retirée(s))")
        return st

    def node_check_final(self, st: EditorialState) -> EditorialState:
        """Contrôle du texte FINAL (après vulgarisation) : la dernière étape n'est pas relue par le relecteur."""
        from transfia.rag.verify import lexical_support, split_with_citations

        def nums(t: str) -> set[str]:
            t = re.sub(r"\[\d+\]", " ", t)                       # numéros de citation [n] : pas des quantités
            found = {n.replace(" ", "") for n in re.findall(r"\d+(?:[.,]\d+)?\s*%?", t)}
            # les ANNÉES (1900-2099) sont des dates de références, pas des quantités (vu : fausse alerte 2012, 2021, 2024)
            return {n for n in found if not re.fullmatch(r"(?:19|20)\d{2}", n)}

        lost = nums(st.draft) - nums(st.final)
        if lost:
            st.final_warnings.append(f"chiffres présents dans le texte validé mais PERDUS à la vulgarisation : "
                                     f"{', '.join(sorted(lost)[:6])} (risque de généralisation abusive)")
        texts = [h.chunk.text for h in st.passages]
        weak = [sent for sent, _ in split_with_citations(st.final)
                if len(tokenize(sent)) >= 3 and max((lexical_support(sent, t) for t in texts), default=0) < 0.5]
        if weak:
            st.final_warnings.append(f"{len(weak)} phrase(s) du texte final sans appui clair dans les passages : "
                                     + " ; ".join(f"« {w[:70]} »" for w in weak[:3]))
        st.history.append("contrôle final")
        return st

    def run(self, topic: str, niveau: str, passages: list[Hit]) -> EditorialState:
        with span("editorial.run", topic=topic) as s:
            st = EditorialState(topic, niveau, passages)
            node = "write"
            while node != "end":
                if node == "write":
                    st, node = self.node_write(st), "review"
                elif node == "review":
                    st = self.node_review(st)
                    node = self.route_after_review(st)
                elif node == "prune":
                    st, node = self.node_prune(st), "simplify"
                elif node == "simplify":
                    st, node = self.node_simplify(st), "check"
                elif node == "check":
                    st, node = self.node_check_final(st), "end"
            s.set(rounds=st.rounds, history=st.history)
            return st
