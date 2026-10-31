"""RAG avancé (F5), codé from scratch pour voir chaque étape.

    question
      -> garde-fous d'entrée (F12)
      -> query rewriting -> [multi-query] -> [HyDE]
      -> recherche hybride + reranking (F4)
      -> seuil de pertinence : rien de pertinent => "je ne trouve pas" (droit de ne pas savoir)
      -> réordonnancement "lost in the middle" (les meilleurs passages aux extrémités du contexte)
      -> prompt (système + passages numérotés + question)
      -> génération (streaming possible)
      -> vérification phrase par phrase (F5) + garde-fous de sortie
      -> cascade : si la réponse est peu fidèle et qu'un gros modèle existe, on relance (F14)
"""
from __future__ import annotations

from typing import Iterator

from transfia.core.models import Answer, Citation, Hit
from transfia.core.prompts import render
from transfia.core.text import tokenize, truncate_words
from transfia.llm.base import Message
from transfia.observability.tracing import current_trace_id, span
from transfia.rag.query import hyde_passage, multi_queries, rewrite_query
from transfia.rag.verify import CITE_RE, REFUSAL, attribute_citations, verify_answer


def reorder_lost_in_middle(hits: list[Hit]) -> list[Hit]:
    """Les LLM exploitent moins bien le MILIEU d'un long contexte : on place les meilleurs
    passages au début ET à la fin. Ordre d'entrée : du plus pertinent au moins pertinent.
    [1,2,3,4,5] -> [1,3,5,4,2]"""
    front = hits[0::2]
    back = hits[1::2][::-1]
    return front + back


class RAGPipeline:
    def __init__(self, llm, engine, settings, guard=None, router=None, nli=None):
        self.llm = llm
        self.engine = engine
        self.settings = settings
        self.guard = guard
        self.router = router
        self.nli = nli

    # ------------------------------------------------------------ préparation
    def retrieve(self, question: str, filters: dict | None = None, history: list[str] | None = None,
                 k: int | None = None) -> tuple[list[Hit], list[str]]:
        cfg = self.settings.rag
        k = k or self.settings.retrieval.k_final
        # La reformulation sert surtout à rendre AUTONOME une question de suite de conversation
        # ("et quand faut-il le donner ?"). Sans historique, elle risque surtout de déformer la question.
        use_rewrite = cfg.rewrite and (bool(history) or not cfg.rewrite_only_with_history)
        q = rewrite_query(self.llm, question, history) if use_rewrite else question
        extra = multi_queries(self.llm, q) if cfg.multi_query else []
        # Filet de sécurité : la reformulation peut PERDRE un terme clé (vu en vrai : "méthane entérique"
        # est devenu "méthane dans l'assiette animale"). On cherche donc AUSSI avec la question d'origine,
        # et RRF fusionne les deux classements.
        if q.strip().lower() != question.strip().lower():
            extra = [question, *extra]
        vectors = [self.engine.embedder.embed_documents([hyde_passage(self.llm, q)])[0]] if cfg.hyde else []
        hits = self.engine.search(q, k=k, filters=filters, extra_queries=extra, dense_vectors=vectors)
        return hits, [q, *extra]

    def build_messages(self, question: str, hits: list[Hit], style: str = "claire et concise",
                       max_sentences: int = 6, preferences: list[str] | None = None
                       ) -> tuple[list[Message], dict[int, Hit]]:
        ordered = reorder_lost_in_middle(hits) if self.settings.rag.reorder_lost_in_middle else hits
        numbered = {i + 1: h for i, h in enumerate(ordered)}
        passages = []
        for n, h in numbered.items():
            text = self.guard.sanitize_passage(h.chunk.text) if self.guard else h.chunk.text
            passages.append({"n": n, "label": h.chunk.meta.get("label", h.chunk.article_id),
                             "title": h.chunk.meta.get("title", ""), "page": h.chunk.page,
                             "text": truncate_words(text, 260)})
        msgs = [Message("system", render("rag_system", style=style, max_sentences=max_sentences,
                                               preferences=preferences or [])),
                Message("user", render("rag_user", passages=passages, question=question))]
        return msgs, numbered

    @staticmethod
    def citations(text: str, numbered: dict[int, Hit]) -> list[Citation]:
        cited = sorted({int(n) for n in CITE_RE.findall(text)} & set(numbered))
        out = []
        for n in cited:
            c = numbered[n].chunk
            out.append(Citation(n=n, article_id=c.article_id, title=c.meta.get("title", ""),
                                authors=c.meta.get("authors", []), year=c.meta.get("year"), doi=c.meta.get("doi"),
                                page=c.page, excerpt=truncate_words(c.text, 40)))
        return out

    def _below_threshold(self, hits: list[Hit], question: str = "") -> bool:
        """Droit de ne pas savoir : rien de pertinent -> refus, sans même appeler le LLM (frugal).

        DOUBLE CRITÈRE : on refuse si le score du reranker est bas ET si les mots porteurs de la question
        sont absents des meilleurs passages. Pourquoi : un score de cross-encoder n'est PAS une probabilité,
        il bouge avec la FORME de la question (vu : "colostrum" -2,6 ; "le colostrum" -3,6 ; seuil -3).
        Le refus "grippe aviaire" (terme absent du corpus) est conservé."""
        t = self.settings.retrieval.min_score
        if not hits:
            return True
        if t is None or hits[0].score >= t:
            return False
        return not self._terms_covered(question, hits[:3])

    def precheck(self, question: str, hits: list[Hit], unknown: set[str]) -> str | None:
        """Contrôles AVANT le LLM, PARTAGÉS par answer() et stream() (vu en vrai : le chemin streaming de
        l'interface avait sa propre copie, sans l'ancrage ni le blocage : deux chemins qui divergent)."""
        if self._below_threshold(hits, question):
            return "no_context"
        if unknown and (not hits or hits[0].score < self.settings.rag.anchor_max_score):
            return "unknown_subject"
        return None

    def unknown_terms(self, question: str) -> set[str]:
        """Mots porteurs de la question (>= 5 lettres) qui n'apparaissent dans AUCUN passage du corpus."""
        from transfia.retrieval.query import clean_query

        bm25 = getattr(self.engine, "bm25", None)
        if not self.settings.rag.anchor_unknown_terms or bm25 is None or not getattr(bm25, "ids", None):
            return set()
        return {w for w in tokenize(clean_query(question)) if len(w) >= 5 and bm25.df.get(w, 0) == 0}

    @staticmethod
    def _terms_covered(question: str, hits: list[Hit], min_ratio: float = 0.8) -> bool:
        """Les mots porteurs de la question (>= 4 lettres) sont-ils presque tous dans UN des passages ?"""
        from transfia.retrieval.query import clean_query

        terms = {w for w in tokenize(clean_query(question)) if len(w) >= 4}
        if not terms:
            return False
        return any(len(terms & set(tokenize(h.chunk.text))) / len(terms) >= min_ratio for h in hits)

    def _refuse(self, question: str, reason: str, hits=None, queries=None) -> Answer:
        return Answer(question=question, text=REFUSAL if reason == "no_context" else reason, refused=True,
                      refusal_reason=reason, hits=hits or [], queries=queries or [], trace_id=current_trace_id())

    # ---------------------------------------------------------------- réponse
    def answer(self, question: str, filters: dict | None = None, history: list[str] | None = None,
               style: str = "claire et concise", preferences: list[str] | None = None) -> Answer:
        """``preferences`` : faits de mémoire long terme (F8), pour adapter la forme de la réponse."""
        with span("rag.answer", question=question) as s:
            if self.guard:
                g = self.guard.check_input(question, " ".join((history or [])[-4:]))
                if not g.allowed:
                    s.set(guard=g.reason)
                    return self._refuse(question, g.message, queries=[question])
            hits, queries = self.retrieve(question, filters, history)
            unknown = self.unknown_terms(question)
            reason = self.precheck(question, hits, unknown)
            if reason:
                s.set(refused=reason, unknown_terms=sorted(unknown))
                ans = self._refuse(question, "no_context", hits, queries)
                ans.refusal_reason = reason
                if unknown:
                    ans.usage["unknown_terms"] = sorted(unknown)
                return ans
            msgs, numbered = self.build_messages(question, hits, style, preferences=preferences)   # la vraie question, pas la reformulation
            model = self.router.route(question).model if self.router else None
            resp = self.llm.chat(msgs, model=model, max_tokens=self.settings.rag.max_tokens)
            text = resp.content.strip()
            added = 0
            if text:                                       # phrases sans citation -> on cherche leur source
                text, added = attribute_citations(text, {n: h.chunk.text for n, h in numbered.items()})
                s.set(citations_added_post_hoc=added)
            checks = verify_answer(text, {n: h.chunk.text for n, h in numbered.items()},
                                   self.settings.rag.verify, nli=self.nli, llm=self.llm, unknown_terms=unknown)
            ans = Answer(question=question, text=text, citations=self.citations(text, numbered), checks=checks,
                         queries=queries, hits=hits, trace_id=current_trace_id(), model=resp.model,
                         usage={"prompt_tokens": resp.usage.prompt_tokens,
                                "completion_tokens": resp.usage.completion_tokens,
                                "latency_s": round(resp.usage.latency_s, 3)})
            if added:
                ans.usage["citations_added_post_hoc"] = added
            # cascade (F14) : réponse peu fidèle -> on retente avec le gros modèle
            bigger = self.router.escalate(ans.faithfulness) if self.router else None
            if bigger and bigger != resp.model:
                s.set(escalated_to=bigger)
                first_faithfulness, first_model = ans.faithfulness, resp.model
                resp = self.llm.chat(msgs, model=bigger, max_tokens=self.settings.rag.max_tokens)
                ans.usage.update({
                    "escalated": f"{first_model} → {bigger} (fidélité {first_faithfulness:.0%} < 60 %)",
                    "prompt_tokens": ans.usage["prompt_tokens"] + resp.usage.prompt_tokens,
                    "completion_tokens": ans.usage["completion_tokens"] + resp.usage.completion_tokens,
                    "latency_s": round(ans.usage["latency_s"] + resp.usage.latency_s, 3),
                })
                text = resp.content.strip()
                ans.text, ans.model = text, resp.model
                ans.citations = self.citations(text, numbered)
                ans.checks = verify_answer(text, {n: h.chunk.text for n, h in numbered.items()},
                                           self.settings.rag.verify, nli=self.nli, llm=self.llm,
                                           unknown_terms=unknown)
            # Garde-fou de SORTIE : aucune phrase soutenue = hallucination probable -> on ne la montre pas
            if (ans.checks and not any(c.supported for c in ans.checks)
                    and self.settings.rag.on_unsupported == "refuse"):
                s.set(blocked="no_supported_sentence")
                ans.usage["blocked_answer"] = ans.text          # gardée pour la transparence et l'analyse
                ans.text, ans.refused, ans.refusal_reason = REFUSAL, True, "unsupported_answer"
                ans.citations = []
            if self.guard:
                ans = self.guard.check_output(ans)
            s.set(faithfulness=ans.faithfulness, n_citations=len(ans.citations), model=ans.model)
            return ans

    def stream(self, question: str, filters: dict | None = None, history: list[str] | None = None) -> Iterator[dict]:
        """Événements pour le front (SSE) : étapes, tokens, citations, vérification."""
        with span("rag.stream", question=question):
            if self.guard:
                g = self.guard.check_input(question, " ".join((history or [])[-4:]))
                if not g.allowed:
                    yield {"type": "refusal", "text": g.message}
                    return
            yield {"type": "step", "text": "Reformulation et recherche dans le corpus…"}
            hits, queries = self.retrieve(question, filters, history)
            yield {"type": "sources", "queries": queries,
                   "hits": [{"article_id": h.chunk.article_id, "title": h.chunk.meta.get("title"),
                             "score": round(h.score, 3)} for h in hits]}
            unknown = self.unknown_terms(question)
            reason = self.precheck(question, hits, unknown)
            if reason:
                extra = f" (le corpus ne contient aucun passage avec « {', '.join(sorted(unknown))} »)" \
                    if reason == "unknown_subject" else ""
                yield {"type": "refusal", "text": REFUSAL + extra}
                return
            msgs, numbered = self.build_messages(question, hits)
            yield {"type": "step", "text": f"Rédaction à partir de {len(hits)} passages…"}
            parts = []
            for piece in self.llm.stream(msgs, max_tokens=self.settings.rag.max_tokens):
                parts.append(piece)
                yield {"type": "token", "text": piece}
            text = "".join(parts).strip()
            # comme answer() : les phrases sans [n] reçoivent leur source APRÈS coup (sinon, vu dans l'interface :
            # réponse correcte mais non citée -> 0/2 phrases soutenues -> bloquée à tort, régression de maj34)
            text, added = attribute_citations(text, {n: h.chunk.text for n, h in numbered.items()})
            if added:
                yield {"type": "answer", "text": text, "citations_added_post_hoc": added}
            yield {"type": "citations", "citations": [c.model_dump() | {"doi_url": c.doi_url}
                                                      for c in self.citations(text, numbered)]}
            checks = verify_answer(text, {n: h.chunk.text for n, h in numbered.items()}, self.settings.rag.verify,
                                   nli=self.nli, llm=self.llm, unknown_terms=unknown)
            yield {"type": "verification", "checks": [c.model_dump() for c in checks]}
            # Garde-fou de SORTIE, comme answer() : les mots sont déjà affichés, donc on les REMPLACE par le refus.
            if checks and not any(c.supported for c in checks) and self.settings.rag.on_unsupported == "refuse":
                yield {"type": "refusal", "text": REFUSAL + " (réponse du modèle bloquée : aucune phrase soutenue "
                                                   "par les sources)"}
                return
            yield {"type": "done", "trace_id": current_trace_id()}
