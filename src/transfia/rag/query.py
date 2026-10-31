"""Transformations de la requête avant la recherche (F5).

1. QUERY REWRITING : rendre la question autonome et précise (corrige, développe, intègre l'historique).
2. MULTI-QUERY : plusieurs reformulations -> plus de RAPPEL (on fusionne ensuite par RRF).
3. HyDE (Hypothetical Document Embeddings) : le LLM rédige une réponse HYPOTHÉTIQUE ; on cherche
   avec l'embedding de ce texte, qui RESSEMBLE davantage aux passages du corpus qu'une question.
   Risque : si le LLM imagine une réponse fausse, la recherche dérive -> à mesurer (F11).
"""
from __future__ import annotations

from pydantic import BaseModel, Field

from transfia.core.prompts import render
from transfia.llm.base import Message, StructuredOutputError, chat_structured
from transfia.observability.tracing import span


class _Queries(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=6)


def rewrite_query(llm, question: str, history: list[str] | None = None) -> str:
    with span("rag.rewrite") as s:
        out = llm.chat([Message("user", render("rewrite", question=question, history=history or []))],
                       temperature=0.0, max_tokens=80).content.strip().strip('"')
        line = out.splitlines()[0].strip() if out else ""
        # garde-fou : si la reformulation est vide ou délirante (trop longue), on garde l'original
        result = line if 3 <= len(line.split()) <= 3 * len(question.split()) + 15 else question
        s.set(original=question, rewritten=result)
        return result


def multi_queries(llm, question: str, n: int = 3) -> list[str]:
    with span("rag.multi_query", n=n) as s:
        try:
            out, _ = chat_structured(llm, [Message("user", render("multi_query", question=question, n=n))], _Queries,
                                     retries=1, max_tokens=200)
            qs = [q.strip() for q in out.queries if q.strip() and q.strip().lower() != question.lower()][:n]
        except StructuredOutputError:
            qs = []
        s.set(queries=qs)
        return qs


def hyde_passage(llm, question: str) -> str:
    with span("rag.hyde") as s:
        text = llm.chat([Message("user", render("hyde", question=question))], temperature=0.3,
                        max_tokens=180).content.strip()
        s.set(hypothetical=text[:300])
        return text
