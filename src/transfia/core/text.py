"""Petits outils de texte partagés (normalisation, découpage en phrases, tokens)."""
from __future__ import annotations

import re
import unicodedata

# Mots vides français courants : ils n'aident pas la recherche lexicale.
STOPWORDS_FR = set(
    """
    a à au aux avec ce ces cet cette dans de des du elle elles en entre est et eux il ils je la le les
    leur leurs lui ma mais me même mes moi mon ne nos notre nous on ou où par pas pour qu que qui sa
    se ses son sont sur ta te tes toi ton tu un une vos votre vous y été être avoir ont a été sans
    plus moins très aussi ainsi alors donc car comme si cela celui celle ceux celles dont lors après
    avant chez sous vers selon peut peuvent fait faire font tout tous toute toutes autre autres
    quel quelle quels quelles d l n s c j m t qu
    """.split()
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_SENT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÀ-ÖØ-Þ0-9«\"(])")


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")


def normalize(text: str) -> str:
    """Minuscules + sans accents + apostrophes unifiées."""
    return strip_accents(text.lower().replace("’", "'"))


def light_stem(word: str) -> str:
    """Racinisation très légère pour le français (pluriels surtout).

    Volontairement simple : un vrai stemmer (Snowball) ferait mieux, mais on veut comprendre.
    """
    for suffix in ("ements", "ement", "ations", "ation", "eaux", "aux", "s", "x"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            if suffix in ("eaux",):
                return word[: -len(suffix)] + "eau"
            if suffix == "aux":
                return word[:-3] + "al"
            return word[: -len(suffix)]
    return word


def tokenize(text: str, stem: bool = True, stopwords: bool = True) -> list[str]:
    words = _WORD_RE.findall(normalize(text))
    out = []
    for w in words:
        if "'" in w:                       # l'élevage -> elevage
            w = w.split("'", 1)[1]
        if stopwords and (w in STOPWORDS_FR or len(w) < 2):
            continue
        out.append(light_stem(w) if stem else w)
    return out


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    return [s.strip() for s in _SENT_RE.split(text) if s.strip()]


def count_tokens(text: str) -> int:
    """Estimation du nombre de tokens LLM : ~1,3 token par mot en français.

    Suffisant pour dimensionner (11.1). Pour un compte exact, il faut le tokenizer du modèle.
    """
    return int(len(text.split()) * 1.3) + 1


def truncate_words(text: str, n: int) -> str:
    words = text.split()
    return text if len(words) <= n else " ".join(words[:n]) + " […]"
