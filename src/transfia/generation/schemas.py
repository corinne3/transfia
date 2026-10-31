"""Schémas des contenus pédagogiques (F7) : la forme EXACTE attendue du LLM.

Deux niveaux de contrôle :
1. le JSON schema (dérivé automatiquement de ces classes) contraint le décodage ;
2. les ``field_validator`` vérifient des règles MÉTIER que le schéma ne sait pas exprimer
   (4 choix distincts, bonne réponse dans les bornes...). En cas d'échec -> retry avec l'erreur.

Règle d'or : des schémas SIMPLES. Un schéma trop complexe dégrade les petits modèles.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Niveau = Literal["grand_public", "bac_pro", "bts", "master"]
NIVEAUX: dict[str, str] = {
    "grand_public": "grand public (aucun prérequis, phrases très courtes, zéro jargon)",
    "bac_pro": "élèves de bac pro agricole (vocabulaire technique de base expliqué, exemples concrets d'exploitation)",
    "bts": "étudiants de BTS (vocabulaire technique, mécanismes expliqués, chiffres clés)",
    "master": "étudiants de master (mécanismes détaillés, nuances, limites des études)",
}


class QuestionQCM(BaseModel):
    enonce: str = Field(description="La question, claire et sans ambiguïté")
    choix: list[str] = Field(description="Exactement 4 propositions", min_length=4, max_length=4)
    # ORDRE VOLONTAIRE : l'explication AVANT l'index. Le LLM écrit le JSON de gauche à droite :
    # s'il choisit l'index d'abord, il "parie" puis justifie après coup (vu : index C, explication de B).
    # En expliquant d'abord, il raisonne, puis l'index découle de son raisonnement ("reason before answer").
    explication: str = Field(description="Pourquoi la bonne proposition est juste, en 1 ou 2 phrases")
    bonne_reponse: int = Field(description="Index (0 à 3) de la proposition que l'explication justifie", ge=0, le=3)
    source: int = Field(description="Numéro du passage [n] qui justifie la réponse", ge=1)

    @field_validator("choix")
    @classmethod
    def choix_distincts(cls, v: list[str]) -> list[str]:
        if len({c.strip().lower() for c in v}) != len(v):
            raise ValueError("les 4 choix doivent être différents")
        return v


class Quiz(BaseModel):
    titre: str
    niveau: Niveau
    questions: list[QuestionQCM] = Field(min_length=1, max_length=10)


class PointCle(BaseModel):
    texte: str
    source: int = Field(ge=1, description="Numéro du passage [n]")


class Terme(BaseModel):
    terme: str
    definition: str
    source: int = Field(ge=1)


class FicheSynthese(BaseModel):
    titre: str
    niveau: Niveau
    objectifs: list[str] = Field(min_length=1, max_length=5, description="Ce que l'apprenant saura faire")
    points_cles: list[PointCle] = Field(min_length=2, max_length=8)
    vocabulaire: list[Terme] = Field(default_factory=list, max_length=8)
    a_retenir: str = Field(description="Une phrase de conclusion")


class CasPratique(BaseModel):
    titre: str
    niveau: Niveau
    contexte: str = Field(description="Description d'une exploitation fictive et de son problème")
    questions: list[str] = Field(min_length=2, max_length=6)
    elements_de_reponse: list[PointCle] = Field(min_length=2, max_length=8)

    @model_validator(mode="after")
    def assez_de_reponses(self):
        if len(self.elements_de_reponse) < len(self.questions) // 2:
            raise ValueError("il faut au moins un élément de réponse pour deux questions")
        return self


class Glossaire(BaseModel):
    titre: str
    niveau: Niveau
    termes: list[Terme] = Field(min_length=3, max_length=15)


CONTENT_TYPES: dict[str, type[BaseModel]] = {
    "quiz": Quiz, "fiche": FicheSynthese, "cas_pratique": CasPratique, "glossaire": Glossaire,
}
SKILL_FOR_TYPE = {"quiz": "quiz-qcm", "fiche": "fiche-synthese", "cas_pratique": "cas-pratique",
                  "glossaire": "glossaire"}


def cited_sources(content: BaseModel) -> list[int]:
    """Numéros de passages cités par le contenu (champs ``source``), triés, sans doublon."""
    nums: list[int] = []

    def walk(obj):
        if isinstance(obj, BaseModel):
            for name, val in obj:
                if name == "source" and isinstance(val, int):
                    nums.append(val)
                walk(val)
        elif isinstance(obj, list):
            for x in obj:
                walk(x)

    walk(content)
    return sorted(set(nums))


def max_source(content: BaseModel) -> int:
    """Plus grand numéro de passage cité (pour vérifier qu'il existe bien)."""
    nums = cited_sources(content)
    return max(nums) if nums else 0
