"""Graphe de connaissances et GraphRAG (F6).

Construction (deux méthodes, à comparer) :
1. CO-OCCURRENCE (from scratch, sans LLM, très frugal) : deux entités citées dans le même passage
   sont reliées ; le poids de l'arête = nombre de passages communs.
2. TRIPLETS par LLM : (sujet, relation, objet) extraits en sortie structurée, avec une liste
   FERMÉE de relations (sinon le graphe devient un fourre-tout).

Nettoyage : ENTITY RESOLUTION (fusion des doublons : "vaches laitières" = "vache laitière").
Analyse : COMMUNAUTÉS (algorithme de Louvain) = groupes de concepts fortement connectés ;
chaque communauté est résumée (par le LLM, ou par ses concepts principaux).

Recherche :
- LOCALE  : on part des entités de la question et on lit leurs voisins (questions précises, multi-sauts).
- GLOBALE : on interroge les résumés de communautés en MAP-REDUCE (questions de synthèse
  "quels sont les grands enjeux…", que le RAG classique gère mal).
"""
from __future__ import annotations

import re

import itertools
import json
from collections import Counter
from pathlib import Path
from typing import Literal

import networkx as nx
from pydantic import BaseModel, Field

from transfia.core.models import Chunk
from transfia.core.text import light_stem, normalize
from transfia.llm.base import Message, StructuredOutputError, chat_structured
from transfia.observability.tracing import span

RELATIONS = ["cause", "favorise", "reduit", "prevent", "affecte", "ameliore", "mesure", "concerne", "compose"]


class Triplet(BaseModel):
    sujet: str
    relation: Literal["cause", "favorise", "reduit", "prevent", "affecte", "ameliore", "mesure", "concerne", "compose"]
    objet: str


class Triplets(BaseModel):
    triplets: list[Triplet] = Field(default_factory=list, max_length=12)


def canonical(name: str) -> str:
    """Entity resolution simple : minuscules, sans accents, chaque mot au singulier approximatif."""
    words = normalize(name).replace("-", " ").split()
    return " ".join(light_stem(w) for w in words)


def is_truncated(resp) -> bool:
    """Ollama renvoie done_reason = "length" quand la génération s'arrête sur la limite de tokens."""
    raw = getattr(resp, "raw", None) or {}
    return isinstance(raw, dict) and raw.get("done_reason") == "length"


class KnowledgeGraph:
    def __init__(self):
        self.g = nx.Graph()
        self.community_of: dict[str, int] = {}
        self.community_summaries: dict[int, str] = {}

    def __len__(self) -> int:
        return self.g.number_of_nodes()

    # --------------------------------------------------------- construction
    def _node(self, name: str, label: str = "concept") -> str:
        key = canonical(name)
        if key not in self.g:
            self.g.add_node(key, label=label, display=name, count=0)
        self.g.nodes[key]["count"] += 1
        return key

    def add_edge(self, a: str, b: str, relation: str, source: str, la: str = "concept", lb: str = "concept") -> None:
        ka, kb = self._node(a, la), self._node(b, lb)
        if ka == kb:
            return
        if self.g.has_edge(ka, kb):
            e = self.g.edges[ka, kb]
            e["weight"] += 1
            e["relations"][relation] = e["relations"].get(relation, 0) + 1
            if source not in e["sources"]:
                e["sources"].append(source)
        else:
            self.g.add_edge(ka, kb, weight=1, relations={relation: 1}, sources=[source])

    def build_cooccurrence(self, chunks: list[Chunk], min_weight: int = 1) -> "KnowledgeGraph":
        with span("graph.cooccurrence", n_chunks=len(chunks)):
            for c in chunks:
                ents = [(name, label) for label, names in c.meta.get("entities", {}).items() for name in names]
                for (a, la), (b, lb) in itertools.combinations(sorted(set(ents)), 2):
                    self.add_edge(a, b, "co-occurrence", c.article_id, la, lb)
            if min_weight > 1:
                weak = [(u, v) for u, v, d in self.g.edges(data=True) if d["weight"] < min_weight]
                self.g.remove_edges_from(weak)
                self.g.remove_nodes_from([n for n in list(self.g) if self.g.degree(n) == 0])
            return self

    def build_from_llm(self, llm, chunks: list[Chunk], max_chunks: int = 30) -> "KnowledgeGraph":
        """Extraction de triplets par LLM : coûteuse sur CPU -> on limite le nombre de passages."""
        with span("graph.llm_triplets", n=min(len(chunks), max_chunks)):
            for c in chunks[:max_chunks]:
                prompt = ("Extrais les relations factuelles du passage sous forme de triplets (sujet, relation, objet). "
                          f"Relations autorisées : {', '.join(RELATIONS)}. Sujets et objets : groupes nominaux courts.\n\n"
                          f"Passage : {c.text}")
                try:
                    out, _ = chat_structured(llm, [Message("user", prompt)], Triplets, retries=1, max_tokens=500)
                except StructuredOutputError:
                    continue
                for t in out.triplets:
                    self.add_edge(t.sujet, t.objet, t.relation, c.article_id)
            return self

    # ------------------------------------------------------------- analyse
    def detect_communities(self, seed: int = 42) -> dict[int, list[str]]:
        if self.g.number_of_edges() == 0:
            return {}
        comms = nx.community.louvain_communities(self.g, weight="weight", seed=seed)
        self.community_of = {n: i for i, c in enumerate(comms) for n in c}
        return {i: sorted(c, key=lambda n: -self.g.nodes[n]["count"]) for i, c in enumerate(comms)}

    def summarize_communities(self, llm=None, top: int = 8) -> dict[int, str]:
        comms = self.detect_communities()
        for cid, nodes in comms.items():
            names = [self.g.nodes[n]["display"] for n in nodes[:top]]
            if llm is None:
                self.community_summaries[cid] = "Concepts liés : " + ", ".join(names)
            else:
                facts = self.local_facts(nodes[0], hops=1)[:12]
                prompt = ("Résume en 2 phrases le thème commun de ces concepts et relations d'un corpus sur l'élevage.\n"
                          f"Concepts : {', '.join(names)}\nRelations : {'; '.join(facts)}")
                self.community_summaries[cid] = llm.chat([Message("user", prompt)], max_tokens=120).content.strip()
        return self.community_summaries

    # ------------------------------------------------------------ recherche
    def find(self, text: str) -> list[str]:
        """Nœuds dont le nom apparaît dans le texte (entités de la question)."""
        t = canonical(text)
        return sorted([n for n in self.g if n and f" {n} " in f" {t} "], key=len, reverse=True)

    def local_facts(self, entity: str, hops: int = 1) -> list[str]:
        start = canonical(entity)
        if start not in self.g:
            matches = self.find(entity)
            if not matches:
                return []
            start = matches[0]
        dist = nx.single_source_shortest_path_length(self.g, start, cutoff=hops)
        # On garde les arêtes dont une extrémité est à moins de `hops` sauts : avec hops=1, SEULEMENT
        # celles du nœud de départ. (Bug corrigé : prendre toutes les arêtes des voisins renvoyait
        # presque tout le graphe, car "méthane" est relié à 50 des 60 nœuds.)
        inner = [n for n, d in dist.items() if d < hops]
        edges = {tuple(sorted((u, v))): (u, v, d) for u, v, d in self.g.edges(inner, data=True)}
        facts = []
        for u, v, d in sorted(edges.values(), key=lambda e: -e[2]["weight"]):
            if u != start and v == start:
                u, v = v, u                                   # le nœud de départ toujours à gauche
            rel = max(d["relations"], key=d["relations"].get)
            facts.append(f"{self.g.nodes[u]['display']} —{rel}→ {self.g.nodes[v]['display']} "
                         f"(poids {d['weight']}, sources : {', '.join(d['sources'][:3])})")
        return facts

    def community_evidence(self, cid: int, chunks: list, n: int = 3, top: int = 6) -> list[str]:
        """Les passages qui citent le PLUS d'entités de la communauté (le texte qui la fonde)."""
        names = [n_ for n_, c in self.community_of.items() if c == cid]
        names = sorted(names, key=lambda x: -self.g.nodes[x]["count"])[:top]
        scored = []
        for ch in chunks:
            t = f" {canonical(ch.text)} "
            k = sum(f" {e} " in t for e in names)
            if k >= 2:
                scored.append((k, ch))
        scored.sort(key=lambda x: -x[0])
        return [f"({ch.meta.get('label', ch.article_id)}) {' '.join(ch.text.split()[:90])}" for _, ch in scored[:n]]

    def global_answer(self, llm, question: str, chunks: list | None = None, trace: list | None = None) -> str:
        """MAP : chaque communauté (résumé + ses meilleurs passages) produit une réponse partielle ;
        REDUCE : synthèse finale. ``trace`` (liste) reçoit le détail de chaque étape map (diagnostic).

        Vus en vrai (qwen2.5:1.5b) : (1) sans passages, il répond RIEN partout (une liste de mots ne suffit pas) ;
        (2) avec l'échappatoire « réponds RIEN si... », il la prend quand même : un petit modèle choisit
        l'option de sortie facile. Donc : quand il y a des extraits, PAS d'échappatoire ; c'est l'étape
        reduce qui fait le tri."""
        if not self.community_summaries:
            self.summarize_communities(llm)
        if not self.community_of:
            self.detect_communities()
        partials = []
        with span("graph.global_search", n_communities=len(self.community_summaries)):
            for cid, summary in self.community_summaries.items():
                ev = self.community_evidence(cid, chunks) if chunks else []
                if ev:
                    prompt = (f"Thème d'un corpus sur l'élevage : {summary}\nExtraits du corpus :\n- "
                              + "\n- ".join(ev) + f"\n\nQuestion : {question}\n"
                              "À partir de CES extraits uniquement, donne 1 ou 2 éléments de réponse en une ou deux "
                              "phrases, en citant la source entre parenthèses.")
                else:
                    prompt = (f"Thème : {summary}\nQuestion : {question}\n"
                              "Si ce thème aide à répondre, donne 1 ou 2 éléments de réponse ; sinon réponds RIEN.")
                out = llm.chat([Message("user", prompt)], max_tokens=150).content.strip()
                kept = bool(out) and not re.fullmatch(r"\W*rien\W*", out, re.I)
                if kept:
                    partials.append(out)
                if trace is not None:
                    trace.append({"communaute": cid, "extraits": len(ev), "garde": kept, "reponse": out[:160]})
            if not partials:
                return "Le graphe ne contient pas d'éléments pour répondre."
            # REDUCE borné : 5 points maximum, une phrase chacun (vu : réponse coupée à "6. Réduction des",
            # la limite de tokens atteinte au milieu d'une phrase).
            prompt = (f"Question : {question}\nÉléments :\n- " + "\n- ".join(partials) +
                      "\nSynthétise en 5 points MAXIMUM, une phrase par point, sans gras ni titre. "
                      "Garde les sources entre parenthèses. N'ajoute rien qui ne soit dans les éléments.")
            resp = llm.chat([Message("user", prompt)], max_tokens=600)
            text = resp.content.strip()
            if is_truncated(resp):
                text += "\n[… réponse coupée : limite de longueur atteinte]"
            return text

    # ------------------------------------------------------- export / persistance
    def to_force_graph(self, max_nodes: int = 150) -> dict:
        """Format attendu par l'affichage du front (nœuds + liens)."""
        nodes = sorted(self.g.nodes, key=lambda n: -self.g.nodes[n]["count"])[:max_nodes]
        keep = set(nodes)
        return {
            "nodes": [{"id": n, "label": self.g.nodes[n]["display"], "group": self.g.nodes[n]["label"],
                       "community": self.community_of.get(n, -1), "size": self.g.nodes[n]["count"]} for n in nodes],
            "links": [{"source": u, "target": v, "weight": d["weight"],
                       "relation": max(d["relations"], key=d["relations"].get)}
                      for u, v, d in self.g.edges(data=True) if u in keep and v in keep],
        }

    def stats(self) -> dict:
        degrees = Counter(dict(self.g.degree()))
        return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges(),
                "communities": len(set(self.community_of.values())),
                "hubs": [self.g.nodes[n]["display"] for n, _ in degrees.most_common(5)]}

    def save(self, path: Path) -> None:
        data = nx.node_link_data(self.g, edges="links")
        data["community_of"] = self.community_of
        data["community_summaries"] = {str(k): v for k, v in self.community_summaries.items()}
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "KnowledgeGraph":
        data = json.loads(path.read_text(encoding="utf-8"))
        kg = cls()
        kg.community_of = data.pop("community_of", {})
        kg.community_summaries = {int(k): v for k, v in data.pop("community_summaries", {}).items()}
        kg.g = nx.node_link_graph(data, edges="links")
        return kg
