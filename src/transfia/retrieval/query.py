"""Nettoyage de la requête AVANT la recherche (F4).

Cas vécu : "Que dit le corpus sur le colostrum ?" -> meilleur score du reranker -5,7 (passage hors sujet, refus),
alors que "colostrum" -> -2,6 (bon passage). Les mots "que dit le corpus" ne sont dans AUCUN passage :
pour l'embedding et le cross-encoder, ce sont du BRUIT qui dilue le sens de la question.

On retire donc les tournures de conversation ("que dit le corpus sur", "parle-moi de", "selon les articles"...)
UNIQUEMENT pour la recherche. Le LLM, lui, reçoit toujours la question d'origine.
Limite : liste de motifs écrite à la main (fragile, mais transparente et testée). Alternative : la
reformulation par LLM (rewrite), plus souple mais plus lente et qui peut déformer (vu : "assiette animale").
"""
from __future__ import annotations

import re

_SRC = r"(?:le corpus|la base|les? articles?|les sources|la litt[ée]rature|la recherche|les [ée]tudes|inrae|tes documents)"
_ABOUT = r"(?:sur|à propos d[eu']?|au sujet d[eu']?|concernant|de|du|des)"
META_PATTERNS = [
    rf"^(?:que|qu'est-ce que)\s+(?:dit|disent|racontent?|sait|savent)\s+{_SRC}\s+{_ABOUT}\s*",
    rf"^(?:selon|d'après)\s+{_SRC}\s*,?\s*",
    r"^(?:parle|dis|explique|raconte|résume|décris)[- ]moi(?:\s+tout)?\s*(?:ce que|à propos d[eu']?|sur|de|du|des)?\s*",
    r"^(?:peux|pourrais|pourriez|pouvez)[- ](?:tu|vous)\s+(?:me\s+|m')?(?:dire|expliquer|parler|résumer|décrire)\s*"
    r"(?:ce que|de|du|des|sur)?\s*",
    r"^(?:je voudrais|j'aimerais|je veux|je souhaite)\s+(?:savoir|connaître|comprendre|en savoir plus sur)\s*",
    rf"\s*,?\s*(?:dans|selon|d'après)\s+{_SRC}\s*",
]
# CONSIGNES DE TÂCHE (vu : "Construis une séquence de 2 séances sur la réduction du méthane entérique" -> refusé,
# alors que le corpus a un article entier sur le sujet) : on ne garde que le SUJET.
_VERB = (r"(?:construis|construire|cr[ée]e|cr[ée]er|pr[ée]pare|pr[ée]parer|fais|faire|r[ée]dige|r[ée]diger|propose|"
         r"proposer|g[ée]n[èe]re|g[ée]n[ée]rer|[ée]labore|con[çc]ois|organise|monte)(?:[- ]moi|[- ]nous)?")
_THING = (r"(?:s[ée]quences?|s[ée]ances?|quiz|qcm|fiches?(?: de synth[èe]se)?|cas pratiques?|activit[ée]s?|"
          r"glossaires?|questions?|cours|modules?|exercices?|supports?|plans?|tp|td|[ée]valuations?)")
TASK_PATTERN = (rf"^{_VERB}\s+(?:un|une|des|le|la|les|\d+)?\s*(?:\d+\s+)?{_THING}"
                rf"(?:\s+(?:de|d')\s*{_THING})?"                           # "questions de QCM"
                rf"(?:\s+(?:de|d')\s*\d+\s+\w+)?"                        # "de 2 séances", "de 4 questions"
                rf"(?:\s+(?:niveau|pour)\s+(?:des |les |un |une )?[\w\-]+(?:\s+pro)?)?"   # "niveau BTS", "pour bac pro"
                rf"\s+(?:sur|à propos d[eu']?|au sujet d[eu']?|concernant|autour d[eu']?|de|du|des)\s*")
META_PATTERNS.insert(0, TASK_PATTERN)
_RE = [re.compile(p, re.I) for p in META_PATTERNS]


def clean_query(query: str) -> str:
    """Retire les tournures de conversation ; renvoie la requête d'origine si rien d'utile ne reste."""
    q = query.replace("’", "'").strip()
    changed = False
    for rx in _RE:
        q, n = rx.subn(" ", q)
        q, changed = q.strip(), changed or n > 0
    if not changed:
        return query                       # question "normale" : on n'y touche pas
    q = re.sub(r"\s+", " ", q).strip(" ?!.,;:")
    return q if len(q) >= 3 else query
