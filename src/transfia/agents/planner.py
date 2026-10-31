"""Plan-and-Execute + humain dans la boucle (F8).

1. PLANIFIER : le LLM produit un plan STRUCTURÉ (Pydantic) : objectif + étapes.
2. VALIDER  : un humain peut accepter, modifier ou refuser le plan (human-in-the-loop) AVANT
   qu'on dépense du calcul à le réaliser.
3. EXÉCUTER : chaque étape est confiée à un agent ReAct, avec le résumé des étapes précédentes.
4. ASSEMBLER : le LLM rédige le livrable final à partir des résultats.

Comparé à un agent ReAct "libre" : plus prévisible, plus facile à suivre et à déboguer.

Ajouts après un essai réel (qwen2.5:1.5b, "séquence sur le bien-être des poules pondeuses") : livrable rempli
de "[Rédiger ici]", section Sources vide, erreur de fond ("viande de poule" pour des pondeuses), et rien
d'affiché pendant les étapes. D'où :
0. COUVERTURE : avant tout, le corpus parle-t-il du sujet ? Sinon on s'arrête (0 appel LLM gaspillé).
   Les passages trouvés sont la BASE DE PREUVES COMMUNE, donnée à chaque étape et à la synthèse.
5. CONTRÔLE QUALITÉ du livrable : textes à compléter ("[Rédiger ici]"), absence de citations ;
   la section Sources est construite par le CODE à partir des [n] réellement cités (pas par le LLM).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel, Field

from transfia.agents.react import ReActAgent
from transfia.core.prompts import render
from transfia.core.text import truncate_words
from transfia.llm.base import Message, chat_structured
from transfia.observability.tracing import span


class Etape(BaseModel):
    titre: str
    description: str
    outil_suggere: str = ""


class Plan(BaseModel):
    objectif: str
    etapes: list[Etape] = Field(min_length=1, max_length=6)


@dataclass
class PlanRun:
    plan: Plan
    results: list[str] = field(default_factory=list)
    final: str = ""
    approved: bool = True
    covered: bool = True
    warnings: list[str] = field(default_factory=list)


Approver = Callable[[Plan], "Plan | None"]     # renvoie le plan (éventuellement modifié) ou None pour refuser
PLACEHOLDER_RE = re.compile(r"\[(?:r[ée]diger ici|à compl[ée]ter|a completer|todo|\.\.\.|…|xxx)[^\]]*\]", re.I)
CITE_RE = re.compile(r"\[(\d+)\]")


STEP_INSTRUCTIONS = ("\n\nRéalise CETTE étape maintenant, en t'appuyant UNIQUEMENT sur les passages ci-dessus : ils "
                     "contiennent l'information, inutile de chercher ailleurs. Cite [n] après chaque information. "
                     "4 à 8 phrases ou une courte liste. Si une information manque, écris « non couvert par le corpus ».")


def check_deliverable(text: str, n_passages: int) -> list[str]:
    """Contrôle qualité du livrable final (règles simples, sans LLM)."""
    out = []
    holes = PLACEHOLDER_RE.findall(text)
    if holes:
        out.append(f"livrable INCOMPLET : {len(holes)} passage(s) à compléter (ex. « {holes[0]} »).")
    cited = {int(n) for n in CITE_RE.findall(text)}
    if n_passages and not cited:
        out.append("aucune citation [n] : le contenu n'est relié à aucune source.")
    elif any(n > n_passages for n in cited):
        out.append("citation d'un passage inexistant.")
    return out


class PlanAndExecute:
    def __init__(self, llm, registry, skills=None, approver: Approver | None = None, max_steps_per_task: int = 5,
                 rag=None, on_step: Callable[[int, int, str, str | None], None] | None = None, k_evidence: int = 6,
                 executor: str = "auto"):
        # EXÉCUTANT d'une étape : "agent" (ReAct avec outils) ou "llm" (1 appel, avec les passages fournis).
        # "auto" = "llm" dès qu'on a une base de preuves. Vu en vrai : avec des outils, le 1,5B relance des
        # recherches maladroites, conclut "je n'ai pas trouvé"... alors que les passages sont DANS son prompt.
        # Quand l'information est déjà là, un appel simple bat un agent (et coûte 1 appel au lieu de 5).
        self.executor = executor
        self.rag = rag                  # pour le contrôle de couverture et la base de preuves (optionnel)
        self.on_step = on_step          # affichage de la progression (étape i/n, titre, résultat)
        self.k_evidence = k_evidence
        self.llm = llm
        self.registry = registry
        self.skills = skills
        self.approver = approver
        self.max_steps = max_steps_per_task

    def plan(self, task: str) -> Plan:
        with span("planner.plan"):
            prompt = render("planner", task=task, tools=self.registry.describe(),
                            skills=self.skills.catalog() if self.skills else "")
            plan, _ = chat_structured(self.llm, [Message("user", prompt)], Plan, retries=2, max_tokens=800)
            return plan

    def evidence(self, task: str):
        """Couverture + base de preuves : (passages numérotés, texte pour les prompts) ou None si hors corpus."""
        hits, _ = self.rag.retrieve(task, k=self.k_evidence)
        if self.rag._below_threshold(hits, task):
            return None
        numbered = {i + 1: h for i, h in enumerate(hits)}
        text = "\n\n".join(f"[{n}] {h.chunk.meta.get('label', h.chunk.article_id)} — {h.chunk.meta.get('title', '')}\n"
                            f"{truncate_words(h.chunk.text, 150)}" for n, h in numbered.items())
        return numbered, text

    def run(self, task: str) -> PlanRun:
        with span("planner.run", task=task) as s:
            numbered, evidence = {}, ""
            if self.rag is not None:
                ev = self.evidence(task)
                if ev is None:
                    msg = ("Le corpus ne contient pas assez d'informations sur ce sujet : plan non lancé "
                           "(aucun appel LLM dépensé). Ajoute des articles sur ce thème, ou reformule.")
                    return PlanRun(Plan(objectif=task, etapes=[Etape(titre="—", description="—")]),
                                   approved=False, covered=False, final=msg)
                numbered, evidence = ev
            plan = self.plan(task)
            if self.approver is not None:
                approved = self.approver(plan)
                if approved is None:
                    return PlanRun(plan, approved=False, final="Plan refusé par l'utilisateur.")
                plan = approved
            run = PlanRun(plan)
            for i, step in enumerate(plan.etapes, 1):
                if self.on_step:
                    self.on_step(i, len(plan.etapes), step.titre, None)
                context = "\n".join(f"- Étape {j}: {r[:600]}" for j, r in enumerate(run.results, 1))
                sub_task = (f"Objectif global : {plan.objectif}\nÉtape {i}/{len(plan.etapes)} : {step.titre} — "
                            f"{step.description}\n" + (f"Résultats des étapes précédentes :\n{context}" if context else "")
                            + (f"\n\nPassages du corpus (base de preuves : appuie-toi dessus et cite [n]) :\n{evidence}"
                               if evidence else ""))
                # pas de garde-fou d'ancrage par étape : une étape "Rédiger" travaille sur les résultats
                # des étapes précédentes, pas sur le corpus (l'étape "Chercher" fournit les sources).
                mode = self.executor if self.executor != "auto" else ("llm" if evidence else "agent")
                if mode == "llm":
                    resp = self.llm.chat([Message("user", sub_task + STEP_INSTRUCTIONS)], max_tokens=500)
                    run.results.append(resp.content.strip())
                else:
                    agent = ReActAgent(self.llm, self.registry, self.skills, max_steps=self.max_steps,
                                       require_grounding=False)
                    run.results.append(agent.run(sub_task).answer)
                if self.on_step:
                    self.on_step(i, len(plan.etapes), step.titre, run.results[-1])
            final_prompt = render("planner_final", task=task, plan=plan.model_dump(), results=run.results,
                                  evidence=evidence)
            final = self.llm.chat([Message("user", final_prompt)], max_tokens=1500).content.strip()
            final = re.sub(r"^```(?:markdown|md)?\s*|\s*```$", "", final).strip()     # retire l'enrobage ```markdown
            if numbered:
                run.warnings = check_deliverable(final, len(numbered))
                cited = sorted({int(n) for n in CITE_RE.findall(final)} & set(numbered))
                if cited:          # Sources construites par le CODE : impossible d'inventer ou d'oublier une source
                    final += "\n\n## Sources\n" + "\n".join(
                        f"[{n}] {numbered[n].chunk.meta.get('label', '')} — {numbered[n].chunk.meta.get('title', '')}"
                        + (f" — https://doi.org/{numbered[n].chunk.meta['doi']}" if numbered[n].chunk.meta.get("doi") else "")
                        for n in cited)
            run.final = final
            s.set(n_steps=len(plan.etapes), warnings=run.warnings)
            return run
