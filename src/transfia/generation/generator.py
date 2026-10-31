"""Générateur de contenus pédagogiques (F7).

Étapes :
1. RETRIEVAL : passages pertinents sur le thème (F4), ou passages d'articles choisis.
2. SKILL : chargement des instructions du type de contenu (progressive disclosure).
3. PROMPT : template Jinja2 = consignes de niveau + skill + exemples (few-shot) + passages numérotés.
4. SORTIE STRUCTURÉE : JSON schema -> objet Pydantic validé, retry avec l'erreur si besoin.
5. CONTRÔLES : chaque ``source`` renvoie à un passage existant ; lisibilité adaptée au niveau ;
   pour un QCM, la bonne réponse est cohérente avec l'explication (consistency.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel

from transfia.core.models import Hit
from transfia.core.prompts import render
from transfia.core.text import tokenize, truncate_words
from transfia.generation.readability import readability_report
from transfia.generation.consistency import blocking, check_quiz, flawed_indices
from transfia.generation.schemas import CONTENT_TYPES, NIVEAUX, SKILL_FOR_TYPE, cited_sources, max_source
from transfia.llm.base import Message, StructuredOutputError, chat_structured
from transfia.observability.tracing import span


@dataclass
class GenerationResult:
    kind: str
    niveau: str
    content: BaseModel
    passages: list[Hit]
    attempts: int
    readability: dict
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        cited = set(cited_sources(self.content))
        return {
            "kind": self.kind, "niveau": self.niveau, "content": self.content.model_dump(),
            "attempts": self.attempts, "readability": self.readability, "warnings": self.warnings,
            "sources": [{"n": i + 1, "article_id": h.chunk.article_id, "title": h.chunk.meta.get("title"),
                         "label": h.chunk.meta.get("label"), "doi": h.chunk.meta.get("doi"), "page": h.chunk.page,
                         "cited": i + 1 in cited}
                        for i, h in enumerate(self.passages)],
        }


def content_to_text(content: BaseModel) -> str:
    """Aplatit le contenu en texte (pour mesurer la lisibilité)."""
    parts: list[str] = []

    def walk(obj):
        if isinstance(obj, BaseModel):
            for _, v in obj:
                walk(v)
        elif isinstance(obj, list):
            for x in obj:
                walk(x)
        elif isinstance(obj, str) and len(obj.split()) > 2:
            parts.append(obj if obj.rstrip().endswith((".", "?", "!")) else obj + ".")

    walk(content)
    return " ".join(parts)


# Consigne système des exemples de fine-tuning (F10). UNE seule définition, partagée par
# finetuning/build_dataset.py (entraînement) et le générateur (utilisation) : le contrat ne peut pas diverger.
FINETUNE_QCM_SYSTEM = ("Tu crées des QCM pédagogiques en élevage à partir d'un passage scientifique. "
                       "Réponds uniquement en JSON : titre, niveau, questions[enonce, choix(4), bonne_reponse, "
                       "explication, source].")
FINETUNED_CONTRACTS = {"transfia-qcm": {"kind": "quiz", "system": FINETUNE_QCM_SYSTEM}}


def check_item_sources(content: BaseModel, hits: list[Hit], threshold: float = 0.35) -> list[str]:
    """Vérification de FOND des éléments sourcés (glossaire, points clés, éléments de réponse).
    Vu dans l'interface : un glossaire « colostrum » de 15 termes, TOUS cités [1], dont « 3-NOP », « méthane »,
    « engraissement »… absents du passage [1] et définis de mémoire. Le schéma ne voit rien : on vérifie
    que le TERME est dans sa source et que la DÉFINITION y est soutenue (recouvrement lexical)."""
    from transfia.rag.verify import lexical_support

    out = []
    items = []
    c = content.model_dump()
    for key in ("termes", "vocabulaire"):
        items += [("terme", t["terme"], t["definition"], t["source"]) for t in c.get(key) or []]
    for key in ("points_cles", "elements_de_reponse"):
        items += [("point", "", p["texte"], p["source"]) for p in c.get(key) or []]
    bad_terms, weak = [], []
    for kind, term, text, n in items:
        if not 1 <= n <= len(hits):
            continue
        passage = hits[n - 1].chunk.text
        if kind == "terme" and term and not set(tokenize(term)) <= set(tokenize(passage)):
            bad_terms.append(term)
        elif lexical_support(text, passage) < threshold:
            weak.append((term or text)[:40])
    if bad_terms:
        out.append(f"{len(bad_terms)} terme(s) ABSENT(S) de leur source (définis de mémoire ?) : {', '.join(bad_terms[:8])}")
    if weak:
        out.append(f"{len(weak)} élément(s) peu soutenu(s) par leur source : {', '.join(weak[:5])}")
    return out


def format_sources(sources: list[dict]) -> list[str]:
    """Sources RÉELLEMENT citées, regroupées par article : "[1][3] Gondret et al., 2026 — titre — DOI".
    (Avant : les 5 passages listés, même non cités, et le même article répété 5 fois.)"""
    used = [s for s in sources if s.get("cited", True)]
    groups: dict[str, list[dict]] = {}
    for s in used:
        groups.setdefault(s.get("article_id") or s.get("title") or str(s["n"]), []).append(s)
    lines = []
    for items in groups.values():
        s = items[0]
        nums = "".join(f"[{x['n']}]" for x in items)
        pages = sorted({x["page"] for x in items if x.get("page")})
        lines.append(f"{nums} {s.get('label')} — {s.get('title')}"
                     + (f" — p. {', '.join(map(str, pages))}" if pages else "")
                     + (f" — https://doi.org/{s['doi']}" if s.get("doi") else ""))
    return lines


def to_markdown(kind: str, content: BaseModel, sources: list[dict]) -> str:
    """Export lisible (Markdown) d'un contenu généré, avec la mention IA et les sources."""
    c = content.model_dump()
    lines = [f"# {c.get('titre', kind)}", f"*Niveau : {c.get('niveau')}*", ""]
    if kind == "quiz":
        for i, q in enumerate(c["questions"], 1):
            lines.append(f"**{i}. {q['enonce']}**")
            lines += [f"- {'ABCD'[j]}. {ch}" for j, ch in enumerate(q["choix"])]
            lines.append(f"> Réponse : {'ABCD'[q['bonne_reponse']]} — {q['explication']} [{q['source']}]\n")
    elif kind == "fiche":
        lines += ["## Objectifs"] + [f"- {o}" for o in c["objectifs"]]
        lines += ["", "## Points clés"] + [f"- {p['texte']} [{p['source']}]" for p in c["points_cles"]]
        if c.get("vocabulaire"):
            lines += ["", "## Vocabulaire"] + [f"- **{t['terme']}** : {t['definition']} [{t['source']}]"
                                               for t in c["vocabulaire"]]
        lines += ["", f"**À retenir :** {c['a_retenir']}"]
    elif kind == "cas_pratique":
        lines += ["## Contexte", c["contexte"], "", "## Questions"] + [f"{i}. {q}" for i, q in enumerate(c["questions"], 1)]
        lines += ["", "## Éléments de réponse"] + [f"- {p['texte']} [{p['source']}]" for p in c["elements_de_reponse"]]
    elif kind == "glossaire":
        lines += [f"- **{t['terme']}** : {t['definition']} [{t['source']}]" for t in c["termes"]]
    lines += ["", "## Sources"] + format_sources(sources)
    lines += ["", "_Contenu généré par IA à partir des sources citées : à vérifier avant diffusion._"]
    return "\n".join(lines)


class ContentGenerator:
    def __init__(self, llm, engine, skills=None, examples: dict[str, str] | None = None, repair_rounds: int = 1,
                 model: str | None = None, drop_flawed: bool = True):
        self.repair_rounds = repair_rounds        # 1 appel LLM de plus, seulement si un QCM a un défaut
        self.model = model                        # None = modèle par défaut du client LLM
        self.drop_flawed = drop_flawed            # retirer les questions encore défectueuses après réparation
        self.llm = llm
        self.engine = engine
        self.skills = skills
        self.examples = examples or {}

    def passages_for(self, topic: str, k: int = 5, filters: dict | None = None,
                     article_ids: list[str] | None = None) -> list[Hit]:
        if article_ids:
            f = dict(filters or {})
            hits = []
            for aid in article_ids:
                hits += [h for h in self.engine.search(topic, k=k, filters=f) if h.chunk.article_id == aid]
            return hits[:k] or self.engine.search(topic, k=k, filters=filters)
        return self.engine.search(topic, k=k, filters=filters)

    def generate(self, kind: str, topic: str, niveau: str = "bts", n_questions: int = 4,
                 filters: dict | None = None, article_ids: list[str] | None = None,
                 passages: list[Hit] | None = None) -> GenerationResult:
        if kind not in CONTENT_TYPES:
            raise ValueError(f"Type inconnu : {kind}. Choix : {', '.join(CONTENT_TYPES)}")
        model_cls = CONTENT_TYPES[kind]
        with span("generate", kind=kind, niveau=niveau, topic=topic) as s:
            hits = passages if passages is not None else self.passages_for(topic, filters=filters, article_ids=article_ids)
            if not hits:
                raise ValueError("Aucun passage trouvé sur ce thème : impossible de générer un contenu sourcé.")
            skill_name = SKILL_FOR_TYPE[kind]
            skill_text = self.skills.load(skill_name) if self.skills and skill_name in self.skills else ""
            contract = FINETUNED_CONTRACTS.get((self.model or "").split(":")[0])
            if contract and kind == contract["kind"]:
                # CONTRAT DE PROMPT : un modèle fine-tuné s'appelle EXACTEMENT comme à l'entraînement
                # (même consigne système, 1 seul passage, même format). Vu en vrai : avec le prompt habituel
                # (skill + exemple + 5 passages), transfia-qcm a RECOPIÉ les choix de l'exemple de style
                # (« l'eau est propre et accessible »…) : décalage de distribution, pire que le modèle de base.
                hits = hits[:1]
                base_msgs = [Message("system", contract["system"]),
                             Message("user", f"Niveau : {niveau}\nPassage [1] : {hits[0].chunk.text}")]
                s.set(prompt_contract=self.model)
            else:
                prompt = render("generate", kind=kind, niveau=niveau, niveau_desc=NIVEAUX[niveau], topic=topic,
                                n_questions=n_questions, skill=skill_text, example=self.examples.get(kind, ""),
                                passages=[{"n": i + 1, "label": h.chunk.meta.get("label", ""),
                                           "text": truncate_words(h.chunk.text, 220)} for i, h in enumerate(hits)])
                base_msgs = [Message("user", prompt)]
            content, attempts = chat_structured(self.llm, base_msgs, model_cls, retries=2,
                                                temperature=0.2, max_tokens=1500, model=self.model)
            warnings = []
            if kind == "quiz":                       # JSON valide ≠ contenu juste : contrôle sémantique
                warnings += check_quiz(content, fix=True)
                # BOUCLE DE RÉPARATION (critique -> correction, comme un relecteur humain) :
                # on renvoie au LLM SON quiz + la liste précise des problèmes, et on garde la meilleure version.
                for _ in range(self.repair_rounds):
                    problems = blocking(warnings)
                    if not problems:
                        break
                    feedback = ("Voici ton quiz :\n" + content.model_dump_json() + "\n\nUn contrôle automatique a "
                                "trouvé ces problèmes :\n- " + "\n- ".join(problems) + "\n\nRéécris le quiz complet. "
                                "Chaque question porte sur une notion DIFFÉRENTE, avec ses propres propositions, et "
                                "UNE SEULE proposition est justifiée par l'explication (les 3 autres sont fausses "
                                "d'après les passages). Garde les questions sans problème.")
                    try:
                        fixed, n = chat_structured(self.llm, base_msgs + [Message("user", feedback)],
                                                   model_cls, retries=1, temperature=0.2, max_tokens=1500,
                                                   model=self.model)
                    except StructuredOutputError:
                        break
                    attempts += n
                    new_warnings = check_quiz(fixed, fix=True)
                    if len(blocking(new_warnings)) < len(problems):      # on ne garde que si c'est MIEUX
                        content, warnings = fixed, ["réparé : " + ", ".join(p.split(" ", 1)[0] for p in problems)] + new_warnings
                    else:
                        warnings.append(f"réparation essayée : {len(blocking(new_warnings))} défaut(s) dans la "
                                        "nouvelle version (pas mieux) -> version d'origine gardée.")
                        break
                # DERNIER RECOURS : mieux vaut 1 bonne question que 3 dont 2 fausses (un QCM faux ENSEIGNE l'erreur).
                bad = flawed_indices(warnings)
                if self.drop_flawed and bad and len(bad) < len(content.questions):
                    content.questions = [q for i, q in enumerate(content.questions, 1) if i not in bad]
                    warnings.append(f"retirées car non réparées : {', '.join(f'Q{i}' for i in bad)} "
                                    f"({len(content.questions)} question(s) gardée(s), numéros d'avertissement = avant retrait).")
                elif bad:
                    warnings.append("toutes les questions ont un défaut : quiz gardé, mais À REVOIR entièrement.")
            warnings += check_item_sources(content, hits)
            if max_source(content) > len(hits):
                warnings.append(f"Une source citée [{max_source(content)}] n'existe pas (seulement {len(hits)} passages).")
            read = readability_report(content_to_text(content), niveau)
            if not read["ok"]:
                warnings.append(f"Lisibilité {read['score']} sous la cible ({read['cible_min']}) pour ce niveau.")
            s.set(attempts=attempts, warnings=warnings, readability=read["score"])
            return GenerationResult(kind, niveau, content, hits, attempts, read, warnings)
