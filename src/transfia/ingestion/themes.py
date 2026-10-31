"""Thèmes du corpus et ÉTIQUETAGE FAIBLE (weak supervision, F3).

Sans annotateurs, on peut créer des étiquettes approximatives avec des RÈGLES (mots-clés).
Ces étiquettes bruitées servent à entraîner un classifieur qui, souvent, GÉNÉRALISE mieux que les
règles elles-mêmes (il apprend d'autres indices du texte). Idéalement, un humain valide un
échantillon pour mesurer le bruit.
"""
from __future__ import annotations

from transfia.core.text import normalize

THEMES: dict[str, list[str]] = {
    "nutrition": ["alimentation", "ration", "nutrition", "aliment", "fourrage", "proteine", "energie", "digestib",
                  "ensilage", "amidon", "acides amines", "lysine", "complement"],
    "sante": ["sante", "maladie", "pathologie", "mammite", "boiterie", "infection", "antibiotique", "vaccin",
              "parasit", "immunit", "mortalite", "diarrhee", "veterinaire"],
    "genetique": ["genetique", "genomique", "selection", "heritabilite", "race", "genotype", "croisement", "gene"],
    "bien_etre": ["bien-etre", "comportement", "stress", "douleur", "logement", "ethologie", "confort"],
    "environnement": ["environnement", "methane", "emission", "carbone", "climat", "azote", "biodiversite",
                      "effluent", "empreinte"],
    "reproduction": ["reproduction", "fertilite", "gestation", "insemination", "velage", "chaleurs", "mise bas"],
    "economie": ["economie", "economique", "revenu", "cout", "prix", "marche", "filiere", "rentabilite"],
    "qualite_produits": ["qualite du lait", "qualite de la viande", "carcasse", "tendrete", "fromage", "oeuf",
                         "composition du lait", "qualite des produits"],
}


def weak_label(text: str) -> str | None:
    """Thème dont les mots-clés apparaissent le plus ; None si aucun."""
    t = normalize(text)
    scores = {theme: sum(t.count(k) for k in kws) for theme, kws in THEMES.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None


def weak_labels(text: str, min_count: int = 2) -> list[str]:
    t = normalize(text)
    return [theme for theme, kws in THEMES.items() if sum(t.count(k) for k in kws) >= min_count]
