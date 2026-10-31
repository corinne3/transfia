"""Lisibilité d'un texte en français (F7) : vérifier que le niveau demandé est respecté.

Formule de Flesch adaptée au français par Kandel et Moles (1958) :
    score = 207 − 1,015 × (mots / phrases) − 73,6 × (syllabes / mots)
Plus le score est ÉLEVÉ, plus le texte est FACILE (≈ 60-70 : grand public ; < 30 : très technique).
Le comptage des syllabes est une heuristique (groupes de voyelles) : un ordre de grandeur, pas une vérité.
"""
from __future__ import annotations

import re

from transfia.core.text import split_sentences, strip_accents

_VOWELS = re.compile(r"[aeiouy]+")

TARGETS = {"grand_public": 60, "bac_pro": 50, "bts": 35, "master": 20}


def count_syllables(word: str) -> int:
    w = strip_accents(word.lower())
    w = re.sub(r"[^a-z]", "", w)
    if not w:
        return 0
    w = re.sub(r"(?<=[^aeiouy])es?$", "", w) or w      # e muet final : "vache" = 1 syllabe
    return max(1, len(_VOWELS.findall(w)))


def flesch_kandel_moles(text: str) -> float:
    sentences = split_sentences(text) or [text]
    words = re.findall(r"[A-Za-zÀ-ÿ'-]+", text)
    if not words:
        return 0.0
    syl = sum(count_syllables(w) for w in words)
    return round(207 - 1.015 * len(words) / len(sentences) - 73.6 * syl / len(words), 1)


def readability_report(text: str, niveau: str) -> dict:
    score = flesch_kandel_moles(text)
    target = TARGETS.get(niveau, 40)
    return {"score": score, "cible_min": target, "ok": score >= target - 10,
            "interpretation": "facile" if score >= 60 else "moyen" if score >= 40 else "difficile"}
