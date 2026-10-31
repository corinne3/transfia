"""Assemblage de l'application (injection de dépendances).

Toutes les briques sont construites ICI à partir de la configuration, et chargées
PARESSEUSEMENT (seulement quand on en a besoin) : utile sur 8 Go de RAM, où l'on ne veut pas
charger le reranker si on ne fait que de la recherche BM25.
"""
from __future__ import annotations

import sys
from functools import cached_property
from pathlib import Path

from transfia.config import Settings, load_settings
from transfia.observability.tracing import configure_tracer


class Lazy:
    """Proxy paresseux : l'objet réel n'est construit qu'au premier accès à l'un de ses attributs."""

    def __init__(self, factory):
        self._factory = factory
        self._obj = None

    def __getattr__(self, name):
        if self._obj is None:
            self._obj = self._factory()
        return getattr(self._obj, name)

    @property
    def loaded(self) -> bool:
        return self._obj is not None


class AppContext:
    def __init__(self, settings: Settings | None = None, llm=None):
        self.settings = settings or load_settings()
        self._llm_override = llm
        tracer = configure_tracer(self.settings.path("traces", "traces.jsonl") if self.settings.observability.traces
                                  else None, enabled=True)
        if self.settings.observability.otel:
            tracer.enable_otel(self.settings.observability.otel_endpoint)

    # ------------------------------------------------------------------- LLM
    @cached_property
    def llm(self):
        if self._llm_override is not None:
            return self._llm_override
        from transfia.llm.factory import build_llm

        return build_llm(self.settings)

    def available(self, model: str | None) -> bool:
        """Le modèle est-il installé ? (Ollama seulement ; ailleurs on suppose que oui.)"""
        if not model:
            return False
        if self.settings.llm.provider != "ollama":
            return True
        if "_ollama_models" not in self.__dict__:
            from transfia.llm.ollama import OllamaClient

            self._ollama_models = OllamaClient(base_url=self.settings.llm.base_url).list_models()
        return any(m == model or m.split(":")[0] == model or m == f"{model}:latest" for m in self._ollama_models)

    def judge_llms(self) -> list:
        """Un client par juge (jury de modèles de familles différentes, F11) — seulement les modèles installés."""
        if self._llm_override is not None:
            return [self._llm_override]
        from transfia.llm.factory import build_llm

        judges = [m for m in self.settings.eval.judges if self.available(m)]
        missing = set(self.settings.eval.judges) - set(judges)
        if missing:
            print(f"⚠️  Juges non installés, ignorés : {', '.join(sorted(missing))} (ollama pull ...)", file=sys.stderr)
        return [build_llm(self.settings, model=m) for m in judges or [self.settings.llm.model]]

    @cached_property
    def router(self):
        from transfia.llm.router import ModelRouter

        big = self.settings.llm.big_model if self.settings.llm.routing else None
        if big and self._llm_override is None and not self.available(big):
            print(f"ℹ️  Gros modèle {big} non installé : routage désactivé (ollama pull {big} pour l'activer).",
                  file=sys.stderr)
            big = None
        return ModelRouter(self.settings.llm.model, big)

    # ------------------------------------------------------------- recherche
    @cached_property
    def embedder(self):
        from transfia.embeddings.embedders import build_embedder

        return build_embedder(self.settings)

    @cached_property
    def store(self):
        from transfia.retrieval.vector_store import build_vector_store

        return build_vector_store(self.settings, lambda: self.embedder.dim)

    @cached_property
    def reranker(self):
        from transfia.retrieval.rerank import build_reranker

        return build_reranker(self.settings)

    @cached_property
    def engine(self):
        from transfia.ingestion.pipeline import load_bm25
        from transfia.retrieval.search import SearchEngine

        r = self.settings.retrieval
        # Lazy : les modèles ne sont chargés qu'au moment où une recherche en a VRAIMENT besoin
        # (une recherche BM25 sans reranking ne charge aucun modèle : moins de RAM, démarrage instantané).
        return SearchEngine(self.store, load_bm25(self.settings), Lazy(lambda: self.embedder),
                            Lazy(lambda: self.reranker), r.k_dense, r.k_bm25, r.rrf_k, r.rerank_top)

    def reload_bm25(self) -> None:
        from transfia.ingestion.pipeline import load_bm25

        if "engine" in self.__dict__:
            self.engine.bm25 = load_bm25(self.settings)

    # ------------------------------------------------------------- ingestion
    def indexer(self, log=print):
        from transfia.ingestion.ner import GazetteerNER
        from transfia.ingestion.parse import build_parser
        from transfia.ingestion.pipeline import Indexer

        ner = GazetteerNER()
        if self.settings.ingestion.ner == "gliner":
            from transfia.ingestion.ner import GlinerNER

            ner = GlinerNER()
        elif self.settings.ingestion.ner == "llm":
            from transfia.ingestion.ner import LLMNER

            ner = LLMNER(self.llm)
        return Indexer(self.settings, self.embedder, self.store, build_parser(self.settings), ner, log=log)

    def index_samples(self, log=print) -> dict:
        from transfia.ingestion.collect import load_sample_articles
        from transfia.ingestion.parse import TextParser

        path = self.settings.resolve("data/samples/sample_corpus.jsonl")
        items = [(a, TextParser().parse_sections(a.id, secs)) for a, secs in load_sample_articles(path)]
        stats = self.indexer(log).run(items)
        self.reload_bm25()
        return stats

    def ensure_index(self, log=print) -> None:
        """Si l'index est vide, indexe le corpus d'exemple (démo hors ligne)."""
        if self.store.count() == 0:
            log("Index vide : indexation du corpus d'exemple (data/samples)…")
            self.index_samples(log)

    # ------------------------------------------------------- briques métier
    @cached_property
    def guard(self):
        from transfia.guardrails import Guard

        g = self.settings.guardrails
        return Guard(g.enabled, g.min_citation_ratio)

    @cached_property
    def nli(self):
        if self.settings.rag.verify != "nli":
            return None
        from transfia.rag.verify import NLIVerifier

        return NLIVerifier()

    @cached_property
    def rag(self):
        from transfia.rag.pipeline import RAGPipeline

        return RAGPipeline(self.llm, self.engine, self.settings, guard=self.guard, router=self.router, nli=self.nli)

    @cached_property
    def skills(self):
        from transfia.skills_runtime.registry import SkillRegistry

        return SkillRegistry(self.settings.resolve(self.settings.skills_dir))

    @cached_property
    def generator(self):
        import json

        from transfia.generation.generator import ContentGenerator

        examples = {}
        ex = self.settings.resolve(self.settings.skills_dir) / "quiz-qcm" / "exemples.json"
        if ex.exists():
            examples["quiz"] = json.dumps(json.loads(ex.read_text(encoding="utf-8")), ensure_ascii=False)
        gen_model = self.settings.llm.generation_model
        if gen_model and not self.available(gen_model):
            print(f"[TransfIA] modèle de génération {gen_model} non installé (ollama pull {gen_model}) : "
                  f"on utilise {self.settings.llm.model}.")
            gen_model = None
        return ContentGenerator(self.llm, self.engine, self.skills, examples, model=gen_model)

    @cached_property
    def kg(self):
        from transfia.graph.kg import KnowledgeGraph

        path = self.settings.path("index", "graph.json")
        return KnowledgeGraph.load(path) if path.exists() else KnowledgeGraph()

    def tools(self):
        from transfia.agents.corpus_tools import build_corpus_tools

        return build_corpus_tools(self.engine, self.skills, self.generator, self.kg if len(self.kg) else None)

    def agent(self, mode: str = "native", memory_facts: list[str] | None = None):
        from transfia.agents.react import ReActAgent

        return ReActAgent(self.llm, self.tools(), self.skills, mode=mode, memory_facts=memory_facts,
                          fallback=self._rag_fallback)

    def _rag_fallback(self, question: str, preferences: list[str] | None = None) -> tuple[str, str]:
        """Plan B de l'agent : le pipeline RAG (F5), avec ses citations, sa vérification et son droit au refus.
        Aussi exposé par le serveur MCP (outil ask_corpus). ``preferences`` : la mémoire long terme de
        l'utilisateur SUIT le passage de relais (sinon elle se perd : vu en vrai)."""
        ans = self.rag.answer(question, preferences=preferences)
        if ans.refused:
            why = {"no_context": "aucun passage assez pertinent", "unsupported_answer": "réponse non soutenue, bloquée"}
            return ans.text, f"refus : {why.get(ans.refusal_reason, ans.refusal_reason)}"
        groups: dict[str, list[int]] = {}             # un article = une ligne : "[1][5] Lagarde et al., 2026"
        for c in ans.citations:
            label = (f"{c.authors[0]} et al., {c.year or 's.d.'}" if c.authors else c.title[:40])
            groups.setdefault(label, []).append(c.n)
        sources = "; ".join("".join(f"[{n}]" for n in ns) + f" {label}" for label, ns in groups.items())
        faith = f"{ans.faithfulness:.0%}" if ans.faithfulness is not None else "n/a"
        pref = " ; préférences appliquées" if preferences else ""
        return (ans.text + (f"\n\nSources : {sources}" if sources else ""),
                f"{len(ans.citations)} source(s) citée(s), fidélité {faith}{pref}")

    def editorial(self):
        from transfia.agents.multi_agent import EditorialWorkflow

        reviewer = None
        judges = self.settings.eval.judges
        other = next((m for m in judges if m != self.settings.llm.model and self.available(m)), None)
        if other and self._llm_override is None:
            from transfia.llm.factory import build_llm

            reviewer = build_llm(self.settings, model=other)
        return EditorialWorkflow(self.llm, reviewer, skills=self.skills)

    @cached_property
    def memory(self):
        from transfia.agents.memory import LongTermMemory

        return LongTermMemory(self.settings.path("memory", "long_term.json"))

    @property
    def data_dir(self) -> Path:
        return self.settings.path(".keep").parent
