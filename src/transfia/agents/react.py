"""Agent ReAct codé from scratch (F8) : Raisonner -> Agir -> Observer, en boucle.

    messages = [system, user]
    tant que pas fini et étapes < max :
        réponse = LLM(messages, tools=schémas)
        si la réponse contient des appels d'outils :
            exécuter chaque outil ; ajouter le résultat aux messages (role="tool")
        sinon :
            réponse finale -> fin

Deux modes :
- "native" : le modèle utilise le TOOL CALLING natif (champ tool_calls structuré).
- "text"   : ReAct "à l'ancienne", par le texte (Thought / Action / Action Input / Observation).
  Utile pour les modèles sans tool calling, et pour comprendre ce que le natif automatise.

Garde-fous : nombre max d'étapes, détection des appels répétés, timeout par outil (registre),
et ANCRAGE : une réponse finale donnée sans avoir consulté le corpus est renvoyée au modèle UNE fois
("cherche d'abord") ; s'il insiste, le CODE lance lui-même search_corpus(question) et lui donne
les passages (plan B : l'agent redevient un workflow) ; sans cet outil, réponse marquée "ungrounded".
Vu en vrai : après le rappel, le 1,5B a répondu "Je vais vérifier la source"... sans appeler d'outil.
Vu en vrai : qwen2.5:1.5b répond "de mémoire" que le méthane vient des "bactéries" de l'estomac.
(Faux : ce sont des ARCHÉES méthanogènes, dans le rumen.) Le prompt l'interdisait déjà :
une consigne ne suffit pas, il faut un contrôle dans le code.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterator

from transfia.core.prompts import render
from transfia.llm.base import Message, ToolCall
from transfia.observability.tracing import span


@dataclass
class Step:
    kind: str                       # "thought" | "tool" | "final" | "error"
    content: str = ""
    tool: str | None = None
    arguments: dict | None = None
    observation: str | None = None


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    stopped_reason: str = "final"

    @property
    def tools_used(self) -> list[str]:
        return [s.tool for s in self.steps if s.kind == "tool"]


_ACTION_RE = re.compile(r"Action\s*:\s*(\w+)\s*\n\s*Action Input\s*:\s*(\{.*?\})\s*(?:\n|$)", re.S)
_FINAL_RE = re.compile(r"(?:Final Answer|Réponse finale)\s*:\s*(.*)", re.S)


# outils qui apportent des CONNAISSANCES du corpus (load_skill apporte un savoir-faire, pas des faits)
GROUNDING_TOOLS = {"search_corpus", "read_article", "generate_content", "graph_neighbors", "ask_corpus"}
NUDGE = ("Tu as répondu sans consulter le corpus. Règle : jamais de réponse de mémoire sur un fait scientifique. "
         "Appelle d'abord search_corpus avec les mots-clés de la question, puis réponds en citant les sources.")


PLAN_B_PROMPT = ("Passages du corpus :\n{obs}\n\nQuestion : {task}\n\nRéponds en 2 à 4 phrases en t'appuyant "
                 "UNIQUEMENT sur ces passages (pas sur tes connaissances), et cite les sources (auteur, année). "
                 "Si les passages ne répondent pas, dis-le.")


def _clean_final(text: str) -> str:
    """Retire un reste de format ReAct (Thought/Action...) d'une réponse finale (vu : "✅ Thought: ... Action: ...")."""
    m = _FINAL_RE.search(text)
    if m:
        return m.group(1).strip()
    kept = [ln for ln in text.splitlines() if not re.match(r"\s*(Thought|Action|Action Input)\s*:", ln)]
    # S'il ne reste RIEN, ce n'était pas une réponse mais un plan d'actions : on renvoie "" (et l'appelant
    # bascule en plan B). Vu en vrai : on rendait alors le texte d'origine, un plan Thought/Action brut.
    return "\n".join(kept).strip()


class ReActAgent:
    def __init__(self, llm, registry, skills=None, max_steps: int = 8, mode: str = "native",
                 observation_chars: int = 2500, memory_facts: list[str] | None = None, model: str | None = None,
                 require_grounding: bool = True, force_search: bool = True, fallback=None):
        # fallback(question, préférences) -> (réponse, résumé) : le workflow de secours (en pratique : le pipeline RAG)
        self.fallback = fallback
        self.require_grounding = require_grounding
        self.force_search = force_search          # plan B : le code cherche à la place de l'agent
        self.llm = llm
        self.registry = registry
        self.skills = skills
        self.max_steps = max_steps
        self.mode = mode
        self.observation_chars = observation_chars
        self.memory_facts = memory_facts or []
        self.model = model

    def system_prompt(self) -> str:
        return render("agent_system", tools=self.registry.describe(), mode=self.mode,
                      skills=self.skills.catalog() if self.skills else "", memory=self.memory_facts)

    def run(self, task: str) -> AgentResult:
        steps: list[Step] = []
        answer, reason = "", "max_steps"
        for event in self.stream(task):
            if event["type"] == "step":
                steps.append(event["step"])
            elif event["type"] == "final":
                answer, reason = event["text"], event["reason"]
        return AgentResult(answer, steps, reason)

    def _can_plan_b(self) -> bool:
        return self.force_search and bool(self.fallback or {"ask_corpus", "search_corpus"} & set(self.registry.tools))

    def _plan_b(self, task: str, why: str) -> Iterator[dict]:
        """PLAN B : l'agent n'a pas su utiliser ses outils -> on bascule en WORKFLOW.
        ⚠️ Contexte NEUF : dans l'historique, le modèle a déjà écrit sa réponse de mémoire (ou ses erreurs) ;
        il les RECOPIE au lieu de lire les passages (vu en vrai)."""
        yield {"type": "step", "step": Step("error", f"{why} -> plan B : le code prend la main (workflow)")}
        if self.fallback:                  # le pipeline RAG complet : citations, vérification, refus
            answer, obs = self.fallback(task, self.memory_facts or None)   # la mémoire SUIT le relais
            yield {"type": "step", "step": Step("tool", tool="pipeline_rag", arguments={"question": task},
                                                observation=obs)}
        elif "ask_corpus" in self.registry.tools:   # mode MCP : le serveur expose le pipeline RAG complet
            args = {"question": task}
            if self.memory_facts:
                args["preferences"] = " | ".join(self.memory_facts)
            raw = self.registry.call("ask_corpus", args)
            answer, sep, summary = raw.rpartition("\n\n(")      # le résumé "(... fidélité ...)" à part
            answer, obs = (answer, "(" + summary) if sep else (raw, "")
            yield {"type": "step", "step": Step("tool", tool="ask_corpus", arguments=args, observation=obs)}
        else:                              # dernier recours : recherche + contexte neuf (sans vérification)
            obs = self.registry.call("search_corpus", {"query": task})[: self.observation_chars]
            yield {"type": "step", "step": Step("tool", tool="search_corpus", arguments={"query": task},
                                                observation=obs)}
            answer = self.llm.chat([Message("user", PLAN_B_PROMPT.format(obs=obs, task=task))],
                                   model=self.model).content.strip()
        yield {"type": "step", "step": Step("final", answer)}
        yield {"type": "final", "text": answer, "reason": "forced_search"}

    def stream(self, task: str) -> Iterator[dict]:
        """Générateur d'événements (pour le front : on affiche chaque étape en direct)."""
        with span("agent.run", task=task, mode=self.mode) as root:
            messages = [Message("system", self.system_prompt()), Message("user", task)]
            seen_calls: dict[str, int] = {}
            # l'ancrage n'a de sens que si le registre contient des outils de connaissance
            must_ground = self.require_grounding and any(n in self.registry.tools for n in GROUNDING_TOOLS)
            grounded, nudged, blocked = False, False, 0
            stop = "max_steps"
            for i in range(self.max_steps):
                with span("agent.step", i=i):
                    if self.mode == "native":
                        resp = self.llm.chat(messages, tools=self.registry.schemas(), model=self.model)
                        calls = resp.tool_calls
                        text = resp.content.strip()
                    else:
                        resp = self.llm.chat(messages, model=self.model)
                        # Le modèle écrit parfois lui-même « Observation : … » (il INVENTE le résultat de l'outil,
                        # vu en vrai). Seul le code a le droit d'écrire une observation : on coupe tout ce qui suit.
                        # (C'est le rôle du "stop sequence" Observation: dans le ReAct d'origine.)
                        text = re.split(r"\n?\s*Observation\s*:", resp.content, maxsplit=1)[0].strip()
                        calls = self._parse_text_actions(text)
                    if not calls:
                        final = _FINAL_RE.search(text)
                        answer = final.group(1).strip() if (final and self.mode == "text") else text
                        if self.mode == "text":
                            answer = _clean_final(answer) or answer   # vu : « ✅ Thought: … Action: … » affiché tel quel
                        if must_ground and not grounded and not nudged:
                            # GARDE-FOU D'ANCRAGE : on refuse la réponse de mémoire, une seule fois
                            nudged = True
                            yield {"type": "step", "step": Step("error", "réponse sans consulter le corpus -> "
                                                                         "renvoyée à l'agent : cherche d'abord")}
                            messages += [Message("assistant", text),
                                         Message("user", NUDGE)]
                            continue
                        if must_ground and not grounded and self._can_plan_b():
                            root.set(n_steps=i + 1, reason="forced_search")
                            yield from self._plan_b(task, "l'agent n'appelle toujours pas d'outil")
                            return
                        reason = "final" if grounded or not must_ground else "ungrounded"
                        step = Step("final", answer)
                        yield {"type": "step", "step": step}
                        root.set(n_steps=i + 1, reason=reason)
                        yield {"type": "final", "text": answer, "reason": reason}
                        return
                    if text:
                        step = Step("thought", text)
                        yield {"type": "step", "step": step}
                    messages.append(Message("assistant", text if self.mode == "text" else resp.content,
                                            tool_calls=calls if self.mode == "native" else []))
                    for call in calls:
                        key = f"{call.name}:{json.dumps(call.arguments, sort_keys=True, ensure_ascii=False)}"
                        seen_calls[key] = seen_calls.get(key, 0) + 1
                        if seen_calls[key] > 2:
                            blocked += 1
                            obs = "ERREUR : cet appel a déjà été fait deux fois avec les mêmes arguments. " \
                                  "Change d'approche ou donne ta réponse finale."
                        else:
                            obs = self.registry.call(call.name, call.arguments)
                            if call.name in GROUNDING_TOOLS and not obs.startswith("ERREUR"):
                                grounded = True
                        t = self.registry.tools.get(call.name)
                        if t is not None and t.return_direct and not obs.startswith("ERREUR"):
                            # le résultat de l'outil est le livrable : on le rend TEL QUEL (structure, sources, avertissements)
                            yield {"type": "step", "step": Step("tool", tool=call.name, arguments=call.arguments,
                                                                observation=obs[:300] + "…")}
                            yield {"type": "step", "step": Step("final", obs)}
                            root.set(n_steps=i + 1, reason="return_direct")
                            yield {"type": "final", "text": obs, "reason": "return_direct"}
                            return
                        obs = obs[: self.observation_chars]
                        step = Step("tool", tool=call.name, arguments=call.arguments, observation=obs)
                        yield {"type": "step", "step": step}
                        if self.mode == "native":
                            messages.append(Message("tool", obs, name=call.name, tool_call_id=call.id))
                        else:
                            messages.append(Message("user", f"Observation : {obs}"))
                    if blocked >= 2:
                        # DÉTECTION DE BOUCLE : le modèle ignore l'avertissement et répète -> on arrête les frais
                        # (vu en vrai : 8 fois le même load_skill fautif, 8 appels LLM pour rien).
                        stop = "stuck"
                        break
            root.set(reason=stop)
            if must_ground and not grounded and self._can_plan_b():
                why = "l'agent tourne en rond" if stop == "stuck" else "nombre maximal d'étapes sans source"
                yield from self._plan_b(task, why)
                return
            # trop d'étapes : on demande une réponse finale SANS outils
            messages.append(Message("user", "Arrête d'utiliser les outils. Donne ta meilleure réponse finale "
                                            "maintenant, en phrases, avec les sources trouvées."))
            final = _clean_final(self.llm.chat(messages, model=self.model).content.strip())
            if not final and self._can_plan_b():          # pas de vraie réponse : plan B plutôt qu'un plan brut
                yield from self._plan_b(task, "aucune réponse finale (seulement des actions)")
                return
            yield {"type": "step", "step": Step("final", final or "(l'agent n'a produit aucune réponse finale)")}
            yield {"type": "final", "text": final, "reason": stop}

    @staticmethod
    def _parse_text_actions(text: str) -> list[ToolCall]:
        out = []
        for name, raw in _ACTION_RE.findall(text):
            try:
                args = json.loads(raw)
            except json.JSONDecodeError:
                args = {"_raw": raw}
            out.append(ToolCall(name=name, arguments=args, id=f"text_{len(out)}"))
        return out[:1]          # ReAct textuel : une action à la fois, puis on observe
