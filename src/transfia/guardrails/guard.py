"""Garde-fous (F12) en entrée et en sortie.

ENTRÉE
- Détection de PROMPT INJECTION (directe : dans la question ; INDIRECTE : cachée dans un document).
- Détection HORS PÉRIMÈTRE : un petit classifieur d'intention (TF-IDF + régression logistique,
  entraîné sur des exemples ci-dessous) combiné à un score de vocabulaire du domaine.
SORTIE
- Taux de citation minimal : une réponse sans sources est signalée.
- Masquage de données personnelles (e-mails, téléphones) : minimisation RGPD.
- Mention de transparence "contenu généré par IA" (AI Act, art. 50).

Attention aux FAUX POSITIFS : des garde-fous trop stricts bloquent des questions légitimes.
On les MESURE (tests + éval de sécurité).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from transfia.core.models import Answer
from transfia.core.text import normalize
from transfia.ingestion.themes import THEMES

INJECTION_PATTERNS = [
    r"ignore[rz]? (toutes |tes |vos |les )?(les )?(instructions|consignes|regles)",
    r"ignore (all |any |the )?(previous|prior|above) (instructions|prompts?)",
    r"oublie (tout|tes instructions|tes consignes)",
    r"(revele|affiche|donne)[- ]?(moi)? (ton|le|tes) (prompt|instructions|system)",
    r"system prompt", r"tu es (maintenant|desormais) ", r"you are now ", r"jailbreak", r"\bdan\b mode",
    r"mode developpeur", r"developer mode", r"<\s*/?\s*(system|passages)\s*>",
]
_INJ = [re.compile(p) for p in INJECTION_PATTERNS]

IN_SCOPE = [
    "quels sont les facteurs de risque des mammites chez la vache laitière",
    "comment réduire les émissions de méthane des ruminants",
    "fais-moi un quiz sur l'alimentation des porcelets",
    "résume les résultats sur le bien-être des poules pondeuses",
    "quel est l'intérêt du pâturage pour les brebis",
    "génère une fiche pour des élèves de BTS sur la reproduction bovine",
    "quelles études parlent de l'antibiorésistance en élevage porcin",
    "explique la sélection génomique des vaches",
    "comment améliorer l'autonomie protéique d'un élevage",
    "prépare un cas pratique sur la santé des veaux",
    "quelle ration pour des chèvres laitières",
    "impact du changement climatique sur l'élevage",
    "comparer les systèmes d'élevage de volailles",
    "qualité du lait et alimentation",
    "quels sont les enjeux de la pisciculture",
    "comment construire une séquence de cours sur la nutrition animale",
]
OUT_SCOPE = [
    "écris-moi un poème d'amour", "donne-moi une recette de tiramisu", "quel temps fera-t-il demain",
    "écris un script python pour trier une liste", "qui va gagner les élections", "raconte une blague",
    "traduis cette phrase en espagnol", "quel est le cours du bitcoin", "aide-moi à écrire une lettre de motivation",
    "résume le dernier film marvel", "comment pirater un compte instagram", "donne-moi des conseils de drague",
    "quel est le meilleur smartphone", "fais mes devoirs de maths", "parle-moi de football",
    "écris une chanson sur la mer",
]

def _gazetteer_terms() -> set[str]:
    from transfia.ingestion.ner import GAZETTEER

    return {form for forms_by_canon in GAZETTEER.values() for forms in forms_by_canon.values() for form in forms}


# Vocabulaire du domaine : thèmes + TOUTES les entités du dictionnaire NER (espèces, pathologies...).
# Sans cela, "traitement contre la grippe aviaire" était rejeté à tort : un FAUX POSITIF classique.
DOMAIN_TERMS = {k for kws in THEMES.values() for k in kws} | _gazetteer_terms() | {
    "elevage", "eleveur", "animal", "animaux", "vache", "porc", "brebis", "chevre", "volaille", "poule", "veau",
    "bovin", "ovin", "caprin", "porcin", "lapin", "poisson", "ruminant", "lait", "viande", "oeuf", "zootechn",
    "quiz", "qcm", "fiche", "cours", "formation", "eleves", "bts", "bac pro", "corpus", "article", "etude",
}

_PII = [(re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[e-mail masqué]"),
        (re.compile(r"(?:(?:\+33|0)\s?[1-9])(?:[\s.-]?\d{2}){4}"), "[téléphone masqué]")]

AI_NOTICE = "Contenu généré par IA à partir des sources citées : à vérifier avant diffusion."


@dataclass
class GuardResult:
    allowed: bool
    reason: str = "ok"
    message: str = ""
    score: float = 1.0


class IntentClassifier:
    """Classifieur hors périmètre : TF-IDF (n-grammes de caractères) + régression logistique."""

    def __init__(self, extra_in: list[str] | None = None, extra_out: list[str] | None = None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        X = IN_SCOPE + (extra_in or []) + OUT_SCOPE + (extra_out or [])
        y = [1] * (len(IN_SCOPE) + len(extra_in or [])) + [0] * (len(OUT_SCOPE) + len(extra_out or []))
        self.model = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), preprocessor=normalize),
                                   LogisticRegression(C=3.0, max_iter=1000, class_weight="balanced"))
        self.model.fit(X, y)

    def proba_in_scope(self, text: str) -> float:
        return float(self.model.predict_proba([text])[0][1])


def domain_score(text: str) -> float:
    t = normalize(text)
    return min(sum(term in t for term in DOMAIN_TERMS) / 2, 1.0)


class Guard:
    def __init__(self, enabled: bool = True, min_citation_ratio: float = 0.5, threshold: float = 0.45):
        self.enabled = enabled
        self.min_citation_ratio = min_citation_ratio
        self.threshold = threshold
        self.intent = IntentClassifier() if enabled else None

    @staticmethod
    def detect_injection(text: str) -> bool:
        t = normalize(text)
        return any(p.search(t) for p in _INJ)

    def check_input(self, text: str, context: str = "") -> GuardResult:
        """``context`` = historique récent : « Et quand faut-il le donner ? » n'a de sens qu'avec lui.
        L'injection est cherchée dans la question seule ; le périmètre, sur question + contexte."""
        if not self.enabled:
            return GuardResult(True)
        if len(text) > 4000:
            return GuardResult(False, "too_long", "Question trop longue (4 000 caractères maximum).", 0.0)
        if self.detect_injection(text):
            return GuardResult(False, "injection", "Cette demande ressemble à une tentative de détournement "
                               "des consignes : je ne peux pas y répondre.", 0.0)
        # combinaison : le vocabulaire du domaine "sauve" une question mal reconnue par le classifieur
        scope_text = f"{context} {text}" if context else text
        score = max(self.intent.proba_in_scope(scope_text), domain_score(scope_text))
        if score < self.threshold:
            return GuardResult(False, "out_of_scope", "Je suis spécialisé dans l'élevage et la formation : "
                               "cette question sort de mon périmètre.", score)
        return GuardResult(True, "ok", "", score)

    def sanitize_passage(self, text: str) -> str:
        """Injection INDIRECTE : on neutralise les lignes d'un document qui ressemblent à des consignes."""
        lines = []
        for line in text.splitlines():
            lines.append("[ligne retirée : consigne suspecte dans le document]" if self.detect_injection(line) else line)
        return "\n".join(lines)

    @staticmethod
    def redact_pii(text: str) -> str:
        for pat, repl in _PII:
            text = pat.sub(repl, text)
        return text

    def check_output(self, ans: Answer) -> Answer:
        if not self.enabled or ans.refused:
            return ans
        ans.text = self.redact_pii(ans.text)
        if ans.checks:
            cited_ratio = sum(bool(c.cited) for c in ans.checks) / len(ans.checks)
            if cited_ratio < self.min_citation_ratio:
                ans.usage["warning"] = "Réponse insuffisamment sourcée : à vérifier."
        ans.usage["notice"] = AI_NOTICE
        return ans
