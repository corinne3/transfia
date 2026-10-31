"""Vérification de la réponse contre ses sources (F5, contrôle des hallucinations).

Pour chaque phrase de la réponse :
- on récupère les passages qu'elle cite ([1], [2]...) ;
- on estime si le passage SOUTIENT la phrase.

Trois méthodes :
1. "lexical" (from scratch) : part des mots porteurs de sens de la phrase présents dans le passage.
   Rapide, mais naïf : ne voit pas une négation ("n'augmente pas").
2. "nli" : un modèle d'INFERENCE EN LANGAGE NATUREL décide si le passage IMPLIQUE la phrase
   (entailment / neutral / contradiction). Bien plus robuste.
3. "llm" : un LLM juge (plus coûteux).

Une phrase SANS citation alors qu'elle contient une information = non soutenue.
"""
from __future__ import annotations

import re

from transfia.core.models import SentenceCheck
from transfia.core.text import split_sentences, tokenize

CITE_RE = re.compile(r"\[(\d+)\]")
REFUSAL = "Je ne trouve pas cette information dans le corpus."


_MD = re.compile(r"\*\*|__|^#+\s*|^\s*(?:\d+[.)]|[-*•])\s+", re.M)


def strip_markdown(text: str) -> str:
    """Retire gras, titres et puces ; chaque élément de liste devient une phrase à part."""
    lines = [_MD.sub("", line).strip() for line in text.splitlines()]
    out = []
    for line in lines:
        line = line.rstrip(" :")
        if line and not line.endswith((".", "!", "?")):
            line += "."                        # chaque ligne / élément de liste = une phrase
        out.append(line)
    return " ".join(x for x in out if x)


def split_with_citations(answer: str) -> list[tuple[str, list[int]]]:
    out = []
    for sent in split_sentences(strip_markdown(answer)):
        cited = [int(n) for n in CITE_RE.findall(sent)]
        clean = CITE_RE.sub("", sent).strip()
        if clean:
            out.append((clean, cited))
    # cas fréquent : la citation est mise en tout début de phrase suivante "[2] ..." -> déjà gérée
    return out


def lexical_support(sentence: str, passage: str) -> float:
    s_terms = set(tokenize(sentence))
    if not s_terms:
        return 1.0
    p_terms = set(tokenize(passage))
    # les nombres doivent correspondre EXACTEMENT : un chiffre inventé est une hallucination classique
    nums = set(re.findall(r"\d+(?:[.,]\d+)?", sentence))
    if nums and not nums <= set(re.findall(r"\d+(?:[.,]\d+)?", passage)):
        return 0.0
    return len(s_terms & p_terms) / len(s_terms)


class NLIVerifier:
    def __init__(self, model: str = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"):
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise ImportError('pip install -e ".[nli]"') from exc
        self.pipe = pipeline("text-classification", model=model, device=-1, top_k=None)

    def support(self, sentence: str, passage: str) -> float:
        res = self.pipe({"text": passage[:2000], "text_pair": sentence})
        res = res[0] if isinstance(res, list) and res and isinstance(res[0], list) else res
        scores = {r["label"].lower(): r["score"] for r in res}
        return float(scores.get("entailment", 0.0))


def verify_answer(answer: str, passages: dict[int, str], method: str = "lexical", threshold: float = 0.6,
                  nli: NLIVerifier | None = None, llm=None, unknown_terms: set[str] | None = None) -> list[SentenceCheck]:
    """``unknown_terms`` : mots de la QUESTION absents de TOUT le corpus. Une phrase qui les contient ne peut
    pas venir du corpus : non soutenue, quel que soit le recouvrement lexical (vu en vrai : « le traitement
    contre la grippe aviaire est le mélange d'ensilage de maïs et de ray-grass » passait, car ensilage,
    maïs et ray-grass étaient dans le passage)."""
    if answer.strip() == REFUSAL or method == "none":
        return []
    checks = []
    for sent, cited in split_with_citations(answer):
        if len(tokenize(sent)) < 2:          # "Oui.", titres... : rien à vérifier
            continue
        if unknown_terms and set(tokenize(sent)) & unknown_terms:
            checks.append(SentenceCheck(sentence=sent, cited=cited, supported=False, score=0.0, method="ancrage",
                                        bad_citations=list(cited)))
            continue
        if not cited:
            checks.append(SentenceCheck(sentence=sent, cited=[], supported=False, score=0.0, method=method))
            continue
        best = 0.0
        bad: list[int] = []                    # PRÉCISION des citations : chaque [n] cité est-il justifié ?
        for n in cited:
            p = passages.get(n, "")
            if not p:
                bad.append(n)                  # numéro de passage inexistant
                continue
            if method == "nli" and nli is not None:
                sc = nli.support(sent, p)
            elif method == "llm" and llm is not None:
                sc = _llm_support(llm, sent, p)
            else:
                sc = lexical_support(sent, p)
            best = max(best, sc)
            if sc < threshold / 2:
                bad.append(n)
        checks.append(SentenceCheck(sentence=sent, cited=cited, supported=best >= threshold, score=round(best, 3),
                                    method=method, bad_citations=bad))
    return checks


def _llm_support(llm, sentence: str, passage: str) -> float:
    from transfia.llm.base import Message

    prompt = (f"Passage :\n{passage}\n\nAffirmation : {sentence}\n\n"
              "Le passage permet-il d'affirmer cela ? Réponds par un seul mot : OUI, PARTIEL ou NON.")
    out = llm.chat([Message("user", prompt)], temperature=0.0, max_tokens=5).content.upper()
    return 1.0 if "OUI" in out else 0.5 if "PARTIEL" in out else 0.0


def attribute_citations(answer: str, passages: dict[int, str], threshold: float = 0.6) -> tuple[str, int]:
    """ATTRIBUTION A POSTERIORI : pour chaque phrase SANS citation, on cherche le passage qui la
    soutient le mieux ; si le support dépasse le seuil, on ajoute [n] à la fin de la phrase.

    Technique réelle (post-hoc attribution), utile avec les petits modèles qui oublient de citer.
    Elle est TRAÇABLE : on renvoie le nombre de citations ajoutées, affiché à l'utilisateur.
    Une phrase qu'aucun passage ne soutient reste sans citation, donc signalée "non soutenue".
    """
    if answer.strip() == REFUSAL:
        return answer, 0
    out, added = [], 0
    for sent in split_sentences(strip_markdown(answer)):
        if CITE_RE.search(sent) or len(tokenize(sent)) < 2:
            out.append(sent)
            continue
        best_n, best = None, 0.0
        for n, p in passages.items():
            sc = lexical_support(sent, p)
            if sc > best:
                best_n, best = n, sc
        if best_n is not None and best >= threshold:
            end = len(sent.rstrip(".!? "))
            sent = f"{sent[:end]} [{best_n}]{sent[end:]}"
            added += 1
        out.append(sent)
    return " ".join(out), added
