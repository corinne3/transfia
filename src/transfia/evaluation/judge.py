"""LLM-as-judge et jury de modèles (F11).

- Une GRILLE explicite (exactitude, niveau, utilité) notée de 1 à 5.
- Le juge écrit son RAISONNEMENT AVANT la note (le champ apparaît en premier dans le schéma) :
  la note est alors conditionnée par l'analyse, et plus stable.
- JURY : plusieurs modèles de familles différentes, puis moyenne (ou vote). Mesure de l'accord
  entre juges : s'ils sont en désaccord, la question mérite un regard humain.
- BIAIS CONNUS à tester : préférence pour les réponses LONGUES, pour la PREMIÈRE réponse présentée
  (biais de position), pour sa propre famille de modèles. ``position_bias`` mesure le 2e.
"""
from __future__ import annotations

import statistics
from typing import Literal

from pydantic import BaseModel, Field

from transfia.core.prompts import render
from transfia.llm.base import Message, StructuredOutputError, chat_structured


class JudgeScore(BaseModel):
    raisonnement: str = Field(description="Analyse courte AVANT de noter")
    exactitude: int = Field(ge=1, le=5)
    pertinence: int = Field(ge=1, le=5)
    niveau_adapte: int = Field(ge=1, le=5)

    @property
    def mean(self) -> float:
        return (self.exactitude + self.pertinence + self.niveau_adapte) / 3


class PairwiseVerdict(BaseModel):
    raisonnement: str
    gagnant: Literal["A", "B", "egalite"]


class LLMJudge:
    def __init__(self, llm, model: str | None = None):
        self.llm = llm
        self.model = model or getattr(llm, "model", "judge")

    def score(self, question: str, answer: str, context: str = "", reference: str = "", niveau: str = "bts") -> JudgeScore | None:
        prompt = render("judge", question=question, answer=answer, context=context, reference=reference, niveau=niveau)
        try:
            out, _ = chat_structured(self.llm, [Message("user", prompt)], JudgeScore, retries=1,
                                     max_tokens=400, model=self.model)
            return out
        except StructuredOutputError:
            return None

    def compare(self, question: str, a: str, b: str) -> str:
        prompt = (f"Question : {question}\n\nRéponse A :\n{a}\n\nRéponse B :\n{b}\n\n"
                  "Laquelle est la plus exacte et la plus utile pour un formateur ? Ne te laisse pas influencer "
                  "par la longueur ni par l'ordre de présentation.")
        try:
            out, _ = chat_structured(self.llm, [Message("user", prompt)], PairwiseVerdict, retries=1,
                                     max_tokens=300, model=self.model)
            return out.gagnant
        except StructuredOutputError:
            return "egalite"

    def position_bias(self, question: str, a: str, b: str) -> dict:
        """On présente (A, B) puis (B, A). Un juge sans biais donne le même gagnant."""
        first = self.compare(question, a, b)
        second = self.compare(question, b, a)
        swapped = {"A": "B", "B": "A", "egalite": "egalite"}[second]
        return {"ordre_1": first, "ordre_2_corrige": swapped, "coherent": first == swapped}


class JudgePanel:
    def __init__(self, judges: list[LLMJudge]):
        self.judges = judges

    def score(self, **kwargs) -> dict:
        scores = [(j.model, j.score(**kwargs)) for j in self.judges]
        valid = [(m, s) for m, s in scores if s is not None]
        if not valid:
            return {"mean": None, "per_judge": {}, "spread": None}
        means = [s.mean for _, s in valid]
        return {
            "mean": round(statistics.mean(means), 3),
            "exactitude": round(statistics.mean(s.exactitude for _, s in valid), 3),
            "spread": round(max(means) - min(means), 3),       # désaccord entre juges
            "per_judge": {m: s.model_dump() for m, s in valid},
        }
