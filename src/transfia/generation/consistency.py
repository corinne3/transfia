"""Cohérence d'un QCM généré (F7) : la bonne réponse correspond-elle à l'explication ?

LE PROBLÈME (vu en vrai avec qwen2.5:1.5b) :
    Choix : A. santé  B. réduire les gaz à effet de serre  C. augmenter le lait  D. viande
    "Réponse : C — ... car elle contribue à réduire les émissions de gaz à effet de serre"
Le JSON est PARFAIT (4 choix distincts, index entre 0 et 3) : le schéma et Pydantic ne voient rien.
C'est une erreur SÉMANTIQUE : une sortie structurée valide n'est pas une sortie juste.

LA PARADE (from scratch, sans LLM) : comparer l'explication à chaque proposition.
1. On prend les mots porteurs de sens de chaque choix (lemmes légers, mots >= 4 lettres).
2. On retire ceux déjà présents dans l'énoncé (ils n'aident pas à départager les choix).
3. Score d'un choix = part de ses mots qui apparaissent dans l'explication.
4. Décision :
   - un choix nettement meilleur que celui marqué juste -> on CORRIGE l'index (et on le signale) ;
   - plusieurs choix ex aequo en tête -> AMBIGU (plusieurs bonnes réponses ?) : avertissement ;
   - sinon : cohérent.
Limite : heuristique lexicale (ne comprend pas une paraphrase ni une négation). Un juge LLM ou NLI
ferait mieux, plus cher. D'où la règle : contenu IA = relu par un humain avant diffusion.
"""
from __future__ import annotations

from transfia.core.text import tokenize

LETTERS = "ABCD"


def _terms(text: str) -> set[str]:
    return {w for w in tokenize(text) if len(w) >= 4}


def _match(word: str, bag: set[str]) -> bool:
    """Égalité sur les 4 premières lettres : réduire ~ réduction, prédire ~ prédictions."""
    return any(word[:4] == w[:4] for w in bag)


def choice_scores(enonce: str, choix: list[str], explication: str) -> list[float]:
    stem, expl = _terms(enonce), _terms(explication)
    scores = []
    for c in choix:
        words = {w for w in _terms(c) if not _match(w, stem)}
        scores.append(sum(_match(w, expl) for w in words) / len(words) if words else 0.0)
    return scores


def check_question(q, min_score: float = 0.5, margin: float = 0.25, fix: bool = True) -> str | None:
    """Renvoie un avertissement (ou None). Si ``fix``, corrige ``q.bonne_reponse`` en place."""
    scores = choice_scores(q.enonce, q.choix, q.explication)
    best = max(range(len(scores)), key=lambda j: scores[j])
    if scores[best] < min_score:
        return None                                   # l'explication ne reprend aucun choix : on ne sait pas
    top = [j for j, s in enumerate(scores) if scores[best] - s < 1e-9]
    marked = q.bonne_reponse
    if len(top) > 1:
        return (f"ambiguë : l'explication justifie plusieurs choix ({', '.join(LETTERS[j] for j in top)}) "
                f"; réponse marquée {LETTERS[marked]}. À reformuler.")
    if best != marked and scores[best] - scores[marked] >= margin:
        if fix:
            q.bonne_reponse = best
        return (f"réponse {LETTERS[marked]} incohérente avec l'explication, qui justifie {LETTERS[best]}"
                + (f" -> corrigée en {LETTERS[best]}." if fix else "."))
    return None


def duplicate_questions(quiz) -> list[str]:
    """Deux questions avec les MÊMES propositions = copier-coller du modèle (vu : Q3 = choix de Q2)."""
    seen: dict[frozenset, int] = {}
    out = []
    for i, q in enumerate(quiz.questions, 1):
        key = frozenset(c.strip().lower() for c in q.choix)
        if key in seen:
            out.append(f"Q{i} reprend les mêmes propositions que Q{seen[key]}. À refaire.")
        else:
            seen[key] = i
    return out


def check_quiz(quiz, fix: bool = True) -> list[str]:
    """Contrôle chaque question ; renvoie la liste des avertissements (Q1, Q2...)."""
    out = []
    for i, q in enumerate(quiz.questions, 1):
        w = check_question(q, fix=fix)
        if w:
            out.append(f"Q{i} {w}")
    return out + duplicate_questions(quiz)


def blocking(warnings: list[str]) -> list[str]:
    """Problèmes qu'on ne sait PAS corriger seul (ambiguïté, doublon) -> à faire réécrire par le LLM.
    Une lettre incohérente, elle, est déjà corrigée : pas besoin de rappeler le modèle."""
    return [w for w in warnings if "ambiguë" in w or "reprend" in w]


def flawed_indices(warnings: list[str]) -> list[int]:
    """Numéros (1, 2...) des questions encore défectueuses. Pour un doublon, seule la COPIE est visée."""
    return sorted({int(w.split(" ", 1)[0][1:]) for w in blocking(warnings) if w.startswith("Q")})
