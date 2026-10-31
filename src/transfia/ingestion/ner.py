"""Reconnaissance d'entités nommées (NER, F3) : espèces, races, pathologies, nutriments, enjeux.

Trois approches, de la plus frugale à la plus flexible :
1. ``GazetteerNER`` (from scratch) : un DICTIONNAIRE du domaine + expressions régulières.
   Précis, rapide, explicable ; mais ne connaît que ce qu'on lui a donné (rappel limité).
2. ``GlinerNER`` : modèle GLiNER, NER ZERO-SHOT. Il compare des fragments du texte aux LIBELLÉS
   qu'on lui fournit ("race bovine", "pathologie"...) : aucun ré-entraînement nécessaire.
3. ``LLMNER`` : un LLM avec une sortie structurée (schéma Pydantic). Flexible mais lent.

Les entités servent à FILTRER (espèce) et à construire le GRAPHE de connaissances (F6).
"""
from __future__ import annotations

import re
from collections import defaultdict

from pydantic import BaseModel

from transfia.core.text import normalize

def _forms(names: list[str]) -> dict[str, list[str]]:
    """{forme canonique: [formes normalisées, singulier + pluriel simple]}"""
    out = {}
    for n in names:
        base = normalize(n)
        plural = base if base.endswith(("s", "x")) else base + "s"
        out[n] = sorted({base, plural})
    return out


GAZETTEER: dict[str, dict[str, list[str]]] = {
    "espece": {
        "bovin": ["bovin", "bovins", "vache", "vaches", "veau", "veaux", "genisse", "genisses", "taurillon", "boeuf", "vaches laitieres"],
        "porcin": ["porc", "porcs", "porcin", "porcins", "truie", "truies", "porcelet", "porcelets", "cochon"],
        "ovin": ["ovin", "ovins", "brebis", "agneau", "agneaux", "mouton", "moutons"],
        "caprin": ["caprin", "caprins", "chevre", "chevres", "chevreau", "chevreaux"],
        "volaille": ["volaille", "volailles", "poulet", "poulets", "poule", "poules", "pondeuse", "pondeuses", "dinde", "canard", "canards"],
        "poisson": ["poisson", "poissons", "truite", "truites", "saumon", "saumons", "aquaculture", "pisciculture"],
        "lapin": ["lapin", "lapins", "lapine", "lapines", "cuniculture"],
        "equin": ["cheval", "chevaux", "equin", "equins", "jument", "juments"],
    },
    "race": _forms([
        "Prim'Holstein", "Holstein", "Normande", "Montbéliarde", "Charolaise", "Limousine", "Blonde d'Aquitaine",
        "Lacaune", "Large White", "Landrace", "Piétrain", "Alpine", "Saanen", "Jersiaise", "Salers", "Aubrac"]),
    "pathologie": _forms([
        "mammite", "boiterie", "acidose", "cétose", "fièvre de lait", "diarrhée", "salmonellose",
        "influenza aviaire", "grippe aviaire", "tuberculose", "paratuberculose", "BVD", "coccidiose",
        "peste porcine africaine", "parasitisme", "mortalité", "strongle"]),
    "nutriment": _forms([
        "protéine", "énergie", "fibre", "amidon", "lysine", "méthionine", "calcium", "phosphore", "vitamine D",
        "vitamine E", "oméga-3", "acide gras", "colostrum", "fourrage", "ensilage", "concentré", "lipide"]),
    "enjeu": _forms([
        "bien-être", "bien-être animal", "méthane", "gaz à effet de serre", "émission", "antibiotique",
        "antibiorésistance", "biodiversité", "pâturage", "autonomie protéique", "génétique", "génomique",
        "reproduction", "qualité du lait", "qualité de la viande", "changement climatique", "consanguinité"]),
}


class Entity(BaseModel):
    text: str
    label: str
    canonical: str
    start: int = -1


class GazetteerNER:
    name = "gazetteer"

    def __init__(self, gazetteer: dict[str, dict[str, list[str]]] | None = None):
        self.gaz = gazetteer or GAZETTEER
        # une seule regex par libellé, les formes les plus longues d'abord ("vaches laitieres" avant "vaches")
        self._patterns: list[tuple[str, str, re.Pattern]] = []
        for label, canon_map in self.gaz.items():
            for canonical, forms in canon_map.items():
                for form in set(forms):
                    pat = re.compile(rf"(?<![a-z0-9]){re.escape(form)}(?![a-z0-9])")
                    self._patterns.append((label, canonical, pat))
        # tri GLOBAL par longueur : "bien-etre animal" est essayé avant "bien-etre"
        self._patterns.sort(key=lambda p: len(p[2].pattern), reverse=True)

    def extract(self, text: str) -> list[Entity]:
        norm = normalize(text)
        taken: list[tuple[int, int]] = []
        found: list[Entity] = []
        for label, canonical, pat in self._patterns:
            for m in pat.finditer(norm):
                if any(a < m.end() and m.start() < b for a, b in taken):
                    continue                                    # pas de chevauchement
                taken.append((m.start(), m.end()))
                found.append(Entity(text=text[m.start():m.end()], label=label, canonical=canonical, start=m.start()))
        return sorted(found, key=lambda e: e.start)


class GlinerNER:
    name = "gliner"
    LABELS = ["espèce animale", "race animale", "pathologie", "nutriment", "molécule", "enjeu d'élevage"]

    def __init__(self, model: str = "urchade/gliner_multi-v2.1", threshold: float = 0.5):
        try:
            from gliner import GLiNER
        except ImportError as exc:
            raise ImportError('pip install -e ".[ner]"') from exc
        self.model = GLiNER.from_pretrained(model)
        self.threshold = threshold

    def extract(self, text: str) -> list[Entity]:
        preds = self.model.predict_entities(text, self.LABELS, threshold=self.threshold)
        return [Entity(text=p["text"], label=p["label"], canonical=normalize(p["text"]), start=p["start"]) for p in preds]


class _LLMEntities(BaseModel):
    entites: list[Entity]


class LLMNER:
    name = "llm"

    def __init__(self, llm):
        self.llm = llm

    def extract(self, text: str) -> list[Entity]:
        from transfia.llm.base import Message, chat_structured

        prompt = ("Extrais les entités du texte : espèce, race, pathologie, nutriment, enjeu. "
                  "Pour chacune : text (tel quel), label, canonical (forme normalisée au singulier).\n\n" + text)
        out, _ = chat_structured(self.llm, [Message("user", prompt)], _LLMEntities)
        return out.entites


def summarize_entities(entities: list[Entity]) -> dict[str, list[str]]:
    """{label: [formes canoniques uniques]} -> stocké dans les métadonnées du chunk (filtres)."""
    out: dict[str, list[str]] = defaultdict(list)
    for e in entities:
        if e.canonical not in out[e.label]:
            out[e.label].append(e.canonical)
    return dict(out)
